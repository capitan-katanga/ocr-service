from collections.abc import Callable

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from httpx import Response

from ocr_service.errors import ERROR_TABLE, ErrorCode, OcrError

from .conftest import FakeEngine, jpeg_bytes, pdf_bytes, png_bytes, wait_until

URL = "/api/v1/ocr/extractions"


def post(client: TestClient, data: bytes, content_type: str = "image/png", **form: str) -> Response:
    return client.post(URL, files={"file": ("x.bin", data, content_type)}, data=form)


def test_full_result(client: TestClient) -> None:
    r = post(client, png_bytes(300, 200))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "engine", "engineVersion", "configHash", "model", "durationMs", "meanConfidence",
        "pageCount", "truncated", "pages", "layoutText",
    }  # fmt: skip
    assert body["engine"] == "paddleocr" and body["model"] == "PP-OCRv6_medium"
    assert body["layoutText"] == "Monto  $ 1.234,56"
    assert body["meanConfidence"] == pytest.approx((5 * 0.9 + 10 * 0.8) / 15)
    assert (body["pageCount"], body["truncated"]) == (1, False)
    assert body["pages"] == [
        {
            "number": 1, "width": 300, "height": 200, "rotation": 0,
            "lines": [
                {"text": "Monto", "confidence": 0.9, "bbox": [80, 420, 230, 460]},
                {"text": "$ 1.234,56", "confidence": 0.8, "bbox": [700, 418, 1000, 466]},
            ],
        }
    ]  # fmt: skip
    assert r.headers["X-Correlation-Id"] == r.headers["X-Trace-Id"]


def test_declared_content_type_does_not_matter(client: TestClient) -> None:
    assert post(client, jpeg_bytes(), "text/plain").status_code == 200
    assert post(client, png_bytes(), "application/pdf").status_code == 200


def test_pdf_with_rotation_and_empty_result(client: TestClient, engine: FakeEngine) -> None:
    engine.lines, engine.angle = [], 90
    body = post(client, pdf_bytes(pages=7), "application/pdf", engine="paddleocr").json()
    assert (body["pageCount"], body["truncated"], len(body["pages"])) == (7, True, 5)
    assert body["pages"][0]["rotation"] == 90
    assert body["layoutText"] == "" and body["meanConfidence"] is None


@pytest.mark.parametrize(
    ("data", "form", "status", "code"),
    [
        (b"nope", {}, 415, "UNSUPPORTED_MEDIA_TYPE"),
        (png_bytes(), {"engine": "tesseract"}, 400, "UNKNOWN_ENGINE"),
        (jpeg_bytes()[:5], {}, 422, "CORRUPT_FILE"),
    ],
)
def test_error_responses(client: TestClient, data: bytes, form: dict[str, str], status: int, code: str) -> None:
    r = post(client, data, **form)
    assert r.status_code == status
    assert r.headers["content-type"] == "application/problem+json"
    assert r.json()["code"] == code


def test_missing_file_is_invalid_request_problem_json(client: TestClient) -> None:
    for r in (client.post(URL, data={"engine": "paddleocr"}), client.post(URL, json={"a": 1}), client.post(URL)):
        assert r.status_code == 400
        assert r.headers["content-type"] == "application/problem+json"
        assert r.json()["code"] == "INVALID_REQUEST"
        assert r.json()["type"].endswith("/invalid-request")
        assert r.headers["X-Correlation-Id"]


def test_disabled_engine(make_client: Callable[..., TestClient]) -> None:
    c = make_client(wait_ready=False, ocr_enabled_engines="")
    r = post(c, png_bytes())
    assert r.status_code == 400
    assert r.json()["code"] == "ENGINE_DISABLED"


def test_loading_then_ready(make_client: Callable[..., TestClient], engine: FakeEngine) -> None:
    engine.release_load.clear()
    c = make_client(wait_ready=False)
    assert c.get("/health/live").json() == {"status": "UP"}
    r = c.get("/health/ready")
    assert r.status_code == 503 and r.json() == {"status": "DOWN", "engines": {"paddleocr": "LOADING"}}
    p = post(c, png_bytes())
    assert p.status_code == 503
    assert p.json()["code"] == "ENGINE_NOT_READY" and p.json()["retryable"] is True
    assert p.headers["Retry-After"] == "5"
    engine.release_load.set()
    wait_until(lambda: c.get("/health/ready").status_code == 200)
    assert c.get("/health/ready").json() == {"status": "UP", "engines": {"paddleocr": "UP"}}


def test_engine_failed_to_load_stays_down(make_client: Callable[..., TestClient], engine: FakeEngine) -> None:
    engine.fail_load = True
    c = make_client(wait_ready=False)
    wait_until(lambda: c.get("/health/ready").json()["engines"]["paddleocr"] == "DOWN")
    assert c.get("/health/ready").status_code == 503


def test_engines(client: TestClient) -> None:
    r = client.get("/api/v1/engines")
    assert r.status_code == 200
    body = r.json()
    assert body["defaultEngine"] == "paddleocr"
    [info] = body["engines"]
    assert info.pop("configHash") and len(r.json()["engines"][0]["configHash"]) == 64
    assert info == {
        "name": "paddleocr", "enabled": True, "ready": True, "engineVersion": "fake 1.0",
        "model": "PP-OCRv6_medium", "providesConfidence": True, "providesBbox": True,
    }  # fmt: skip


@pytest.mark.parametrize("code", list(ErrorCode))
def test_every_error_code_maps_to_status_retryable_and_type(client: TestClient, code: ErrorCode) -> None:
    router = APIRouter()

    @router.get(f"/boom/{code.value}")
    async def boom() -> None:
        raise OcrError(code, "detail")

    client.app.include_router(router)
    r = client.get(f"/boom/{code.value}")
    status, retryable = ERROR_TABLE[code]
    body = r.json()
    assert r.status_code == status and r.headers["content-type"] == "application/problem+json"
    assert (body["code"], body["retryable"], body["status"]) == (code.value, retryable, status)
    assert body["type"] == f"https://expense-tracker.local/problems/{code.value.lower().replace('_', '-')}"
    assert ("Retry-After" in r.headers) == (status == 503)
    assert r.headers["X-Correlation-Id"]


def test_unexpected_exception_is_internal_error(client: TestClient) -> None:
    router = APIRouter()

    @router.get("/crash")
    async def crash() -> None:
        raise RuntimeError("secret detail")

    client.app.include_router(router)
    r = client.get("/crash")
    assert r.status_code == 500 and r.headers["content-type"] == "application/problem+json"
    assert r.json()["code"] == "INTERNAL_ERROR" and r.json()["retryable"] is True
    assert "secret" not in r.text
    assert r.headers["X-Correlation-Id"] and r.headers["X-Trace-Id"]


def test_malformed_multipart_is_invalid_request(client: TestClient) -> None:
    r = client.post(URL, content=b"--b\r\nNone--b--\r\n", headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 400 and r.headers["content-type"] == "application/problem+json"
    assert r.json()["code"] == "INVALID_REQUEST"


def test_other_http_errors_keep_the_default_response(client: TestClient) -> None:
    assert client.get("/nope").status_code == 404
