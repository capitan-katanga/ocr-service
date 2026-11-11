from typing import Any

import numpy as np
from PIL import Image

from ocr_service.engines.paddle import PaddleOcrEngine
from ocr_service.settings import Settings


class FakePaddleOCR:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.seen: list[np.ndarray] = []
        self.result: dict[str, Any] = {
            "doc_preprocessor_res": {"output_img": np.zeros((30, 50, 3), dtype=np.uint8)},
            "rec_texts": ["Monto", "$ 1.234,56"],
            "rec_scores": [0.98, 1.7],
            "rec_boxes": np.array([[10.4, 5, 40, 20], [-3, 6, 50.9, 25]]),
        }

    def predict(self, arr: np.ndarray) -> list[dict[str, Any]]:
        self.seen.append(arr)
        return [self.result]


def loaded(**settings: Any) -> tuple[PaddleOcrEngine, FakePaddleOCR]:
    made: list[FakePaddleOCR] = []

    def factory(**kw: Any) -> FakePaddleOCR:
        made.append(FakePaddleOCR(**kw))
        return made[0]

    engine = PaddleOcrEngine(Settings(**settings), factory)
    engine.load()
    return engine, made[0]


def test_constructor_parameters() -> None:
    _, ocr = loaded(ocr_models_dir="/m", ocr_cpu_threads=3, ocr_use_textline_orientation=True)
    kw = ocr.kwargs
    assert kw["engine"] == "paddle" and kw["cpu_threads"] == 3
    assert kw["use_doc_orientation_classify"] is False and kw["use_doc_unwarping"] is False
    assert kw["use_textline_orientation"] is True
    assert "lang" not in kw
    assert kw["text_detection_model_name"] == "PP-OCRv6_medium_det"
    assert kw["text_detection_model_dir"] == "/m/PP-OCRv6_medium_det"
    assert kw["text_recognition_model_dir"] == "/m/PP-OCRv6_medium_rec"
    assert kw["doc_orientation_classify_model_dir"] == "/m/PP-LCNet_x1_0_doc_ori"
    assert kw["doc_unwarping_model_dir"] == "/m/UVDoc"
    assert kw["textline_orientation_model_dir"] == "/m/PP-LCNet_x1_0_textline_ori"


def test_result_mapping() -> None:
    engine, ocr = loaded()
    page = engine.recognize(Image.new("RGB", (60, 40), (255, 0, 0)))
    assert (page.width, page.height, page.angle) == (50, 30, 0)
    assert [ln.text for ln in page.lines] == ["Monto", "$ 1.234,56"]
    assert page.lines[0].bbox == (10, 5, 40, 20)
    assert page.lines[1].bbox == (0, 6, 50, 25)  # sin negativos
    assert page.lines[1].confidence == 1.0  # acotada a [0, 1]
    assert tuple(ocr.seen[0][0, 0]) == (0, 0, 255)  # BGR


def test_angle_from_doc_preprocessor() -> None:
    engine, ocr = loaded()
    ocr.result["doc_preprocessor_res"]["angle"] = 180
    assert engine.recognize(Image.new("RGB", (60, 40))).angle == 180
    ocr.result["doc_preprocessor_res"]["angle"] = -1  # clasificador apagado
    assert engine.recognize(Image.new("RGB", (60, 40))).angle == 0


def test_default_factory_is_paddleocr(monkeypatch: Any) -> None:
    import sys
    import types

    created: list[dict[str, Any]] = []
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCR = lambda **kw: created.append(kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    PaddleOcrEngine(Settings()).load()
    assert created and created[0]["engine"] == "paddle"
