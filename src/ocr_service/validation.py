import io
import math
from typing import Literal

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image, UnidentifiedImageError

from ocr_service.errors import ErrorCode, OcrError
from ocr_service.settings import Settings

MAX_BYTES = 10 * 1024 * 1024
Kind = Literal["image", "pdf"]


def sniff(data: bytes) -> Kind:
    """Tipo por magic bytes; el content type declarado no cuenta."""
    if data.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
        return "image"
    if data.startswith(b"%PDF-"):
        return "pdf"
    raise OcrError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, "Only PNG, JPEG and PDF are supported.")


def open_pdf(data: bytes) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        if e.err_code == pdfium_raw.FPDF_ERR_PASSWORD:
            raise OcrError(ErrorCode.ENCRYPTED_PDF, "The PDF is password protected.") from e
        raise OcrError(ErrorCode.CORRUPT_FILE, "The PDF cannot be parsed.") from e


class InputValidator:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def validate(self, data: bytes) -> Kind:
        if len(data) > MAX_BYTES:
            raise OcrError(ErrorCode.PAYLOAD_TOO_LARGE, "The file is larger than 10 MiB.")
        kind = sniff(data)
        if kind == "image":
            self._check_image(data)
        else:
            self._check_pdf(data)
        return kind

    def _check_bounds(self, width: int, height: int) -> None:
        s = self._s
        if min(width, height) < s.ocr_min_image_side or width * height > s.ocr_max_image_pixels:
            raise OcrError(
                ErrorCode.IMAGE_OUT_OF_BOUNDS,
                f"Image of {width}x{height} px is out of bounds (min side {s.ocr_min_image_side} px).",
            )

    def _check_image(self, data: bytes) -> None:
        try:
            with Image.open(io.BytesIO(data)) as img:
                size = img.size
        except Image.DecompressionBombError as e:
            raise OcrError(ErrorCode.IMAGE_OUT_OF_BOUNDS, "The image has too many pixels.") from e
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            raise OcrError(ErrorCode.CORRUPT_FILE, "The image cannot be decoded.") from e
        self._check_bounds(*size)

    def _check_pdf(self, data: bytes) -> None:
        scale = self._s.ocr_pdf_dpi / 72
        pdf = open_pdf(data)
        try:
            for i in range(min(len(pdf), self._s.ocr_max_pdf_pages)):
                w, h = pdf[i].get_size()
                self._check_bounds(math.ceil(w * scale), math.ceil(h * scale))
        finally:
            pdf.close()
