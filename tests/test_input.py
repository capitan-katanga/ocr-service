import io

import pytest
from PIL import Image

from ocr_service.errors import ErrorCode, OcrError
from ocr_service.preprocessing import Preprocessor
from ocr_service.settings import Settings
from ocr_service.validation import MAX_BYTES, InputValidator

from .conftest import jpeg_bytes, pdf_bytes, png_bytes

ENCRYPTED_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 100 100]>>endobj\n"
    b"4 0 obj<</Filter/Standard/V 1/R 2/P -4/O(" + b"a" * 32 + b")/U(" + b"b" * 32 + b")>>endobj\n"
    b"trailer<</Root 1 0 R/Size 5/Encrypt 4 0 R"
    b"/ID[<00112233445566778899aabbccddeeff><00112233445566778899aabbccddeeff>]>>\n"
    b"%%EOF"
)

s = Settings()
validator = InputValidator(s)
pre = Preprocessor(s)


def code_of(data: bytes) -> ErrorCode:
    with pytest.raises(OcrError) as e:
        validator.validate(data)
    return e.value.code


@pytest.mark.parametrize(("data", "kind"), [(png_bytes(), "image"), (jpeg_bytes(), "image"), (pdf_bytes(), "pdf")])
def test_accepted_types_by_magic_bytes(data: bytes, kind: str) -> None:
    assert validator.validate(data) == kind


@pytest.mark.parametrize("data", [b"GIF89a" + b"0" * 100, b"hello world"])
def test_unsupported_types(data: bytes) -> None:
    assert code_of(data) == ErrorCode.UNSUPPORTED_MEDIA_TYPE


def test_size_limit() -> None:
    ok = png_bytes() + b"\0" * (MAX_BYTES - len(png_bytes()))
    assert len(ok) == MAX_BYTES and validator.validate(ok) == "image"
    assert code_of(ok + b"\0") == ErrorCode.PAYLOAD_TOO_LARGE


def test_corrupt_files() -> None:
    assert code_of(jpeg_bytes()[:5]) == ErrorCode.CORRUPT_FILE
    assert code_of(b"%PDF-1.4 garbage") == ErrorCode.CORRUPT_FILE
    truncated = jpeg_bytes(400, 400)[:-300]
    validator.validate(truncated)  # el header alcanza para validar...
    with pytest.raises(OcrError) as e:
        pre.prepare(truncated, "image")  # ...pero no decodifica
    assert e.value.code == ErrorCode.CORRUPT_FILE


def test_encrypted_pdf() -> None:
    assert code_of(ENCRYPTED_PDF) == ErrorCode.ENCRYPTED_PDF


def test_image_bounds() -> None:
    assert code_of(png_bytes(16, 200)) == ErrorCode.IMAGE_OUT_OF_BOUNDS
    assert code_of(png_bytes(7000, 6000, "L")) == ErrorCode.IMAGE_OUT_OF_BOUNDS  # 42 MP


def test_decompression_bomb_is_out_of_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    assert code_of(png_bytes(100, 100)) == ErrorCode.IMAGE_OUT_OF_BOUNDS


def test_pdf_bounds_use_the_rasterized_size() -> None:
    assert code_of(pdf_bytes(size=(10, 200))) == ErrorCode.IMAGE_OUT_OF_BOUNDS  # 28 px de ancho a 200 dpi
    big = InputValidator(Settings(ocr_max_image_pixels=100_000))
    with pytest.raises(OcrError) as e:
        big.validate(pdf_bytes(size=(100, 200)))  # 278 x 556 px
    assert e.value.code == ErrorCode.IMAGE_OUT_OF_BOUNDS


def test_exif_rotation_is_reported() -> None:
    page = pre.prepare(jpeg_bytes(200, 100, exif_orientation=6), "image").pages[0]
    assert page.rotation == 90 and page.image.size == (100, 200)
    assert pre.prepare(jpeg_bytes(200, 100), "image").pages[0].rotation == 0


def test_image_is_converted_to_rgb() -> None:
    buf = io.BytesIO()
    Image.new("L", (100, 100)).save(buf, "PNG")
    assert pre.prepare(buf.getvalue(), "image").pages[0].image.mode == "RGB"


def test_pdf_truncated_to_max_pages_and_rasterized_at_dpi() -> None:
    prepared = pre.prepare(pdf_bytes(pages=7, size=(100, 200)), "pdf")
    assert (len(prepared.pages), prepared.page_count, prepared.truncated) == (5, 7, True)
    assert [p.number for p in prepared.pages] == [1, 2, 3, 4, 5]
    assert prepared.pages[0].image.width == pytest.approx(100 * s.ocr_pdf_dpi / 72, abs=1)


def test_pdf_not_truncated() -> None:
    prepared = pre.prepare(pdf_bytes(pages=2), "pdf")
    assert (prepared.page_count, prepared.truncated) == (2, False)
