"""Contra la imagen real `ocr-service:local` (docker build -t ocr-service:local .). Corre con `pytest -m slow`."""

import io
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from PIL import Image, ImageDraw, ImageFont
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy

pytestmark = pytest.mark.slow

FONT = Path(__file__).parent / "fixtures/fonts/DejaVuSans.ttf"
# (etiqueta, valor): datos ficticios
ROWS = [
    ("Monto", "$ 1.234,56"),
    ("Fecha", "15/03/2026"),
    ("CUIT", "30-71234567-8"),
    ("Descripción", "Café con leche"),
    ("Destino", "Año nuevo, señal"),
]


def render_receipt() -> Image.Image:
    img = Image.new("RGB", (1100, 140 + 90 * len(ROWS)), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 40), "Comprobante de transferencia", font=ImageFont.truetype(str(FONT), 44), fill="black")
    font = ImageFont.truetype(str(FONT), 38)
    for i, (label, value) in enumerate(ROWS):
        y = 140 + 90 * i
        draw.text((60, y), label, font=font, fill="black")
        draw.text((600, y), value, font=font, fill="black")
    return img


def as_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def as_pdf(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PDF", resolution=150)
    return buf.getvalue()


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    wait = HttpWaitStrategy(8084, "/health/ready").for_status_code(200).with_startup_timeout(300)
    with DockerContainer("ocr-service:local").with_exposed_ports(8084).waiting_for(wait) as c:
        yield f"http://{c.get_container_host_ip()}:{c.get_exposed_port(8084)}"


@pytest.mark.parametrize(
    ("name", "make", "content_type"), [("png", as_png, "image/png"), ("pdf", as_pdf, "application/pdf")]
)
def test_fictional_receipt_keeps_rows_and_accents(base_url: str, name: str, make, content_type: str) -> None:  # type: ignore[no-untyped-def]
    r = httpx.post(
        f"{base_url}/api/v1/ocr/extractions",
        files={"file": (f"receipt.{name}", make(render_receipt()), content_type)},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    text = r.json()["layoutText"]
    rows = [
        row.replace(" ", "") for row in text.splitlines()
    ]  # el reconocedor a veces pierde el espacio de "$ 1.234,56"
    for label, value in ROWS:
        assert any(label in row and value.replace(" ", "") in row for row in rows), f"{label!r} y {value!r}:\n{text}"
    for exact in ("Descripción", "Café", "Año", "señal"):
        assert exact in text
    assert r.json()["meanConfidence"] > 0.8
