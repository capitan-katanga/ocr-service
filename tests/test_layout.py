import pytest

from ocr_service.engines.types import OcrLine
from ocr_service.layout import LayoutReconstructor

lay = LayoutReconstructor()


def line(text: str, x: int, yc: int, h: int = 40, w: int = 100, conf: float | None = 0.9) -> OcrLine:
    return OcrLine(text, conf, (x, yc - h // 2, x + w, yc + h // 2))


def test_tilted_lines_of_one_row_stay_together_and_neighbors_do_not() -> None:
    page = [
        line("Monto", 0, 100),
        line("$ 1.234,56", 400, 112),  # misma fila, algo más abajo por la inclinación
        line("Fecha", 0, 160),  # renglón vecino, a 1,5 alturas
        line("01/02/2026", 400, 172),
    ]
    assert lay.layout_text([page]) == "Monto  $ 1.234,56\nFecha  01/02/2026"


def test_orders_by_x_and_collapses_spaces() -> None:
    page = [line("b", 300, 100), line("a   x", 0, 100)]
    assert lay.layout_text([page]) == "a  x  b"


def test_pages_are_separated_by_blank_line_and_empty_pages_skipped() -> None:
    assert lay.layout_text([[line("uno", 0, 50)], [], [line("dos", 0, 50)]]) == "uno\n\ndos"


def test_lines_without_bbox_use_normalized_engine_text() -> None:
    page = [OcrLine("a   b\t c", None, None), OcrLine("d", None, None)]
    assert lay.layout_text([page]) == "a b c\nd"


def test_blank_text_lines_are_dropped() -> None:
    assert lay.layout_text([[line("  ", 0, 10), line("x", 0, 60)]]) == "x"


def test_mean_confidence_is_weighted_by_chars() -> None:
    pages = [[line("aaaa", 0, 10, conf=1.0)], [line("b", 0, 10, conf=0.0), line("cc", 0, 90, conf=None)]]
    assert lay.mean_confidence(pages) == pytest.approx(4 / 5)


def test_no_lines() -> None:
    assert lay.layout_text([[]]) == ""
    assert lay.mean_confidence([[]]) is None
    assert lay.mean_confidence([[line("x", 0, 0, conf=None)]]) is None
