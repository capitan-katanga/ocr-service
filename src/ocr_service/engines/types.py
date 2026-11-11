from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float | None
    bbox: tuple[int, int, int, int] | None  # [xMin, yMin, xMax, yMax] sobre la imagen que salió del pipeline


@dataclass(frozen=True)
class PageOcr:
    lines: list[OcrLine]
    width: int
    height: int
    angle: int  # rotación extra que aplicó el DocPreprocessor (0 si está apagado)


class OcrEngine(Protocol):
    engine_version: str
    model: str
    provides_confidence: bool
    provides_bbox: bool

    def load(self) -> None: ...

    def recognize(self, image: Image.Image) -> PageOcr: ...
