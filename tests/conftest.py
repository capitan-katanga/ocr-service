import io
import threading
import time
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from PIL import Image

from ocr_service.engines.types import OcrLine, PageOcr
from ocr_service.main import create_app
from ocr_service.settings import Settings
from ocr_service.telemetry import Telemetry

LINES = [
    OcrLine("Monto", 0.9, (80, 420, 230, 460)),
    OcrLine("$ 1.234,56", 0.8, (700, 418, 1000, 466)),
]


class FakeEngine:
    engine_version = "fake 1.0"
    model = "PP-OCRv6_medium"
    provides_confidence = True
    provides_bbox = True

    def __init__(self, lines: list[OcrLine] | None = None, angle: int = 0) -> None:
        self.lines = LINES if lines is None else lines
        self.angle = angle
        self.release_load = threading.Event()
        self.release_load.set()
        self.block_recognize: threading.Event | None = None
        self.fail_load = False
        self.sizes: list[tuple[int, int]] = []

    def load(self) -> None:
        self.release_load.wait(10)
        if self.fail_load:
            raise RuntimeError("no models")

    def recognize(self, image: Image.Image) -> PageOcr:
        self.sizes.append(image.size)
        if self.block_recognize:
            self.block_recognize.wait(10)
        return PageOcr(self.lines, image.width, image.height, self.angle)


class Obs:
    def __init__(self) -> None:
        self.spans = InMemorySpanExporter()
        self.metrics = InMemoryMetricReader()
        self.logs = InMemoryLogRecordExporter()
        tracer = TracerProvider()
        tracer.add_span_processor(SimpleSpanProcessor(self.spans))
        logger = LoggerProvider()
        logger.add_log_record_processor(SimpleLogRecordProcessor(self.logs))
        self.telemetry = Telemetry(tracer, MeterProvider(metric_readers=[self.metrics]), logger)

    def metric(self, name: str) -> list[object]:
        data = self.metrics.get_metrics_data()
        if data is None:
            return []
        return [m.data for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics if m.name == name]


@pytest.fixture
def obs() -> Obs:
    return Obs()


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def make_client(engine: FakeEngine, obs: Obs) -> Iterator[Callable[..., TestClient]]:
    stack: list[TestClient] = []

    def make(wait_ready: bool = True, **settings: object) -> TestClient:
        app = create_app(Settings(**settings), engine, obs.telemetry)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        stack.append(client)
        if wait_ready:
            wait_until(lambda: app.state.registry.ready)
        return client

    yield make
    for c in stack:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def wait_until(cond: Callable[[], bool], timeout: float = 5) -> None:
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timeout"
        time.sleep(0.01)


def png_bytes(w: int = 200, h: int = 100, mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (w, h), "white" if mode == "RGB" else 255).save(buf, "PNG")
    return buf.getvalue()


def jpeg_bytes(w: int = 200, h: int = 100, exif_orientation: int | None = None) -> bytes:
    img = Image.new("RGB", (w, h), "white")
    exif = Image.Exif()
    if exif_orientation:
        exif[0x0112] = exif_orientation
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def pdf_bytes(pages: int = 1, size: tuple[int, int] = (100, 200)) -> bytes:
    imgs = [Image.new("RGB", size, "white") for _ in range(pages)]
    buf = io.BytesIO()
    imgs[0].save(buf, "PDF", save_all=True, append_images=imgs[1:], resolution=72)
    return buf.getvalue()
