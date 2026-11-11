import logging
import threading
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.trace import SpanKind

from ocr_service import main
from ocr_service.healthcheck import main as healthcheck
from ocr_service.settings import Settings
from ocr_service.telemetry import ContextFilter, Telemetry, outcome_of

from .conftest import FakeEngine, Obs, png_bytes

URL = "/api/v1/ocr/extractions"
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
TRACEPARENT = f"00-{TRACE_ID}-00f067aa0ba902b7-01"


def extract(client: TestClient, **headers: str) -> Any:
    return client.post(URL, files={"file": ("a.png", png_bytes(), "image/png")}, headers=headers)


def test_baggage_wins_over_correlation_header(client: TestClient) -> None:
    r = extract(client, baggage="correlationId=from-baggage,userId=u1", **{"X-Correlation-Id": "from-header"})
    assert r.headers["X-Correlation-Id"] == "from-baggage"


def test_correlation_header_without_baggage(client: TestClient) -> None:
    assert extract(client, **{"X-Correlation-Id": "from-header"}).headers["X-Correlation-Id"] == "from-header"


def test_without_any_id_the_trace_id_is_used(client: TestClient) -> None:
    r = extract(client)
    assert r.headers["X-Correlation-Id"] == r.headers["X-Trace-Id"]
    assert len(r.headers["X-Trace-Id"]) == 32


def test_server_span_is_child_of_traceparent(client: TestClient, obs: Obs) -> None:
    r = extract(client, traceparent=TRACEPARENT)
    assert r.headers["X-Trace-Id"] == TRACE_ID
    spans = {s.name: s for s in obs.spans.get_finished_spans()}
    server = next(s for s in spans.values() if s.kind == SpanKind.SERVER)
    assert server.parent is not None and format(server.parent.span_id, "016x") == "00f067aa0ba902b7"
    for name in ("ocr preprocess", "ocr infer", "ocr layout"):
        assert spans[name].attributes["model.name"] == "PP-OCRv6_medium"  # type: ignore[index]
        assert format(spans[name].context.trace_id, "032x") == TRACE_ID


def test_health_is_not_traced_nor_given_trace_id(client: TestClient, obs: Obs) -> None:
    r = client.get("/health/ready")
    assert "X-Trace-Id" not in r.headers and not obs.spans.get_finished_spans()


def test_logs_carry_context_and_hide_content(client: TestClient, obs: Obs, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        extract(client, baggage="correlationId=c-1,userId=u-1", traceparent=TRACEPARENT, Authorization="Bearer s3cr3t")
    records = [r.log_record for r in obs.logs.get_finished_logs() if "POST" in str(r.log_record.body)]
    assert records
    attrs = records[0].attributes
    assert attrs["traceId"] == TRACE_ID and len(attrs["spanId"]) == 16  # type: ignore[arg-type,index]
    assert attrs["correlationId"] == "c-1" and attrs["userId"] == "u-1"  # type: ignore[index]
    text = " ".join(str(r.log_record.body) for r in obs.logs.get_finished_logs()) + caplog.text
    assert "s3cr3t" not in text and "Monto" not in text and "layoutText" not in text
    assert "'authorization': '***'" in text


def test_context_filter_without_span() -> None:
    record = logging.LogRecord("x", logging.INFO, "f", 1, "m", None, None)
    ContextFilter().filter(record)
    assert (record.traceId, record.spanId, record.correlationId, record.userId) == ("", "", "", "")  # type: ignore[attr-defined]


def points(obs: Obs, name: str) -> list[Any]:
    return [p for data in obs.metric(name) for p in data.data_points]  # type: ignore[attr-defined]


def test_duration_outcome_per_status_class_and_error_counter(
    make_client: Callable[..., TestClient], engine: FakeEngine, obs: Obs
) -> None:
    ok = make_client()
    extract(ok)  # 200 success
    ok.post(URL, files={"file": ("a", b"nope", "image/png")})  # 415 rejected
    engine.release_load.clear()
    loading = make_client(wait_ready=False)
    extract(loading)  # 503 unavailable
    engine.release_load.set()
    slow = make_client(ocr_inference_timeout=0.1)
    engine.block_recognize = threading.Event()
    extract(slow)  # 504 timeout
    engine.block_recognize.set()

    by_outcome = {p.attributes["outcome"]: p for p in points(obs, "ocr.extraction.duration")}
    assert set(by_outcome) == {"success", "rejected", "unavailable", "timeout"}
    for p in by_outcome.values():
        assert p.attributes["ocr.engine"] == "paddleocr" and p.attributes["model.name"] == "PP-OCRv6_medium"
        assert list(p.explicit_bounds) == [0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
    ok_attrs = by_outcome["success"].attributes
    assert ok_attrs["model.name"] == "PP-OCRv6_medium" and ok_attrs["ocr.engine"] == "paddleocr"
    errors = {p.attributes["code"]: p.value for p in points(obs, "ocr.extraction.errors")}
    assert errors == {"UNSUPPORTED_MEDIA_TYPE": 1, "ENGINE_NOT_READY": 1, "ENGINE_TIMEOUT": 1}
    assert points(obs, "ocr.extraction.confidence")[0].count == 1
    assert points(obs, "ocr.inference.queued")[0].value == 0


@pytest.mark.parametrize(
    ("status", "outcome"), [(200, "success"), (400, "rejected"), (422, "rejected"), (500, "unavailable"),
                            (503, "unavailable"), (504, "timeout")]
)  # fmt: skip
def test_outcome_classes(status: int, outcome: str) -> None:
    assert outcome_of(status) == outcome


def test_app_is_built_lazily_for_uvicorn(monkeypatch: pytest.MonkeyPatch, obs: Obs) -> None:
    monkeypatch.setattr(Telemetry, "otlp", classmethod(lambda cls, settings: obs.telemetry))
    app = main.app  # type: ignore[attr-defined]
    assert app.title == "ocr-service API"
    with pytest.raises(AttributeError):
        main.nope  # type: ignore[attr-defined]  # noqa: B018


def test_healthcheck_exit_codes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    class Resp:
        def __init__(self, status: int) -> None:
            self.status = status

        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *a: object) -> None:
            return None

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: Resp(200))
    assert healthcheck() == 0
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: Resp(503))
    assert healthcheck() == 1

    def refuse(url: str, timeout: int) -> None:
        raise ConnectionRefusedError

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert healthcheck() == 1


def test_otlp_telemetry_targets_the_configured_endpoint_and_shuts_down() -> None:
    tel = Telemetry.otlp(Settings(otel_exporter_otlp_endpoint="http://127.0.0.1:1/", otel_service_name="svc"))
    assert tel.tracer_provider.resource.attributes["service.name"] == "svc"
    tel.shutdown()
