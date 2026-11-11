import io
from dataclasses import dataclass

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from ocr_service.errors import ErrorCode, OcrError
from ocr_service.settings import Settings
from ocr_service.validation import Kind, open_pdf

EXIF_ORIENTATION = 0x0112
EXIF_ROTATION = {3: 180, 6: 90, 8: 270}

ImageFile.LOAD_TRUNCATED_IMAGES = False


@dataclass(frozen=True)
class PreparedPage:
    number: int
    image: Image.Image
    rotation: int


@dataclass(frozen=True)
class Prepared:
    pages: list[PreparedPage]
    page_count: int
    truncated: bool


class Preprocessor:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def prepare(self, data: bytes, kind: Kind) -> Prepared:
        if kind == "image":
            return Prepared([self._image(data)], 1, False)
        return self._pdf(data)

    def _image(self, data: bytes) -> PreparedPage:
        try:
            with Image.open(io.BytesIO(data)) as img:
                rotation = EXIF_ROTATION.get(img.getexif().get(EXIF_ORIENTATION, 1), 0)
                fixed = ImageOps.exif_transpose(img)
                fixed.load()
                return PreparedPage(1, fixed.convert("RGB"), rotation)
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            raise OcrError(ErrorCode.CORRUPT_FILE, "The image cannot be decoded.") from e

    def _pdf(self, data: bytes) -> Prepared:
        pdf = open_pdf(data)
        try:
            total = len(pdf)
            scale = self._s.ocr_pdf_dpi / 72
            shown = min(total, self._s.ocr_max_pdf_pages)
            pages = [PreparedPage(i + 1, pdf[i].render(scale=scale).to_pil().convert("RGB"), 0) for i in range(shown)]
        finally:
            pdf.close()
        return Prepared(pages, total, total > shown)
