import re
import statistics
from collections.abc import Sequence

from ocr_service.engines.types import OcrLine

LAYOUT_VERSION = 1  # subir cuando cambie la salida de LayoutReconstructor (entra al configHash)


class LayoutReconstructor:
    def layout_text(self, pages: Sequence[Sequence[OcrLine]]) -> str:
        texts = (self._page_text(lines) for lines in pages)
        return "\n\n".join(t for t in texts if t)

    def mean_confidence(self, pages: Sequence[Sequence[OcrLine]]) -> float | None:
        scored = [(len(ln.text), ln.confidence) for lines in pages for ln in lines if ln.confidence is not None]
        chars = sum(n for n, _ in scored)
        if chars == 0:
            return None
        return sum(n * c for n, c in scored) / chars

    def _page_text(self, lines: Sequence[OcrLine]) -> str:
        lines = [ln for ln in lines if ln.text.strip()]
        if any(ln.bbox is None for ln in lines):
            return "\n".join(re.sub(r"\s+", " ", ln.text).strip() for ln in lines)
        rows = [_normalize("  ".join(ln.text.strip() for ln in row)) for row in self._rows(lines)]
        return "\n".join(rows)

    def _rows(self, lines: Sequence[OcrLine]) -> list[list[OcrLine]]:
        if not lines:
            return []
        boxes = {id(ln): ln.bbox for ln in lines if ln.bbox}
        half = statistics.median(b[3] - b[1] for b in boxes.values()) / 2

        def center(ln: OcrLine) -> float:
            b = boxes[id(ln)]
            return (b[1] + b[3]) / 2

        rows: list[list[OcrLine]] = []
        for ln in sorted(lines, key=center):
            # contra la media del renglón, no contra el primero: una línea inclinada no arrastra el corte
            if rows and abs(center(ln) - statistics.fmean(center(x) for x in rows[-1])) < half:
                rows[-1].append(ln)
            else:
                rows.append([ln])
        return [sorted(row, key=lambda x: boxes[id(x)][0]) for row in rows]


def _normalize(text: str) -> str:
    return re.sub(r" {3,}", "  ", re.sub(r"[\t\r\n]+", " ", text))
