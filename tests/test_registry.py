from importlib.metadata import version

import pytest

from ocr_service import layout
from ocr_service.engines import EngineRegistry
from ocr_service.engines.paddle import PaddleOcrEngine
from ocr_service.errors import ErrorCode, OcrError
from ocr_service.settings import Settings

from .conftest import FakeEngine


def registry(**kw: object) -> EngineRegistry:
    return EngineRegistry(Settings(**kw), {"paddleocr": FakeEngine()})


def test_unknown_and_disabled_engines() -> None:
    reg = registry()
    reg.load_all()
    with pytest.raises(OcrError) as e:
        reg.resolve("tesseract")
    assert e.value.code == ErrorCode.UNKNOWN_ENGINE
    off = registry(ocr_enabled_engines="")
    with pytest.raises(OcrError) as e2:
        off.resolve(None)
    assert e2.value.code == ErrorCode.ENGINE_DISABLED
    assert off.ready  # sin motores habilitados no hay nada que esperar


def test_not_ready_until_loaded_and_down_when_load_fails() -> None:
    reg = registry()
    with pytest.raises(OcrError) as e:
        reg.resolve(None)
    assert e.value.code == ErrorCode.ENGINE_NOT_READY
    reg.load_all()
    assert reg.resolve(None)[0] == "paddleocr"
    bad = FakeEngine()
    bad.fail_load = True
    down = EngineRegistry(Settings(), {"paddleocr": bad})
    down.load_all()
    assert down.state == {"paddleocr": "DOWN"} and not down.ready


CHANGES = [
    {"ocr_use_doc_orientation_classify": True},
    {"ocr_use_doc_unwarping": True},
    {"ocr_use_textline_orientation": True},
    {"ocr_text_detection_model": "PP-OCRv6_small_det"},
    {"ocr_text_recognition_model": "PP-OCRv6_small_rec"},
    {"ocr_pdf_dpi": 300},
    {"ocr_max_pdf_pages": 3},
]


def test_config_hash_is_stable_and_sensitive_to_each_option(monkeypatch: pytest.MonkeyPatch) -> None:
    base = registry().config_hash("paddleocr")
    assert base == registry().config_hash("paddleocr")
    assert len(base) == 64
    hashes = {registry(**c).config_hash("paddleocr") for c in CHANGES}
    assert base not in hashes and len(hashes) == len(CHANGES)
    reg = registry()
    monkeypatch.setattr(layout, "LAYOUT_VERSION", layout.LAYOUT_VERSION + 1)
    assert reg.config_hash("paddleocr") != base


def test_engine_version_and_model_use_installed_packages() -> None:
    engine = PaddleOcrEngine(Settings())
    assert engine.engine_version == (
        f"paddleocr {version('paddleocr')}; paddlepaddle {version('paddlepaddle')}; "
        "PP-OCRv6_medium_det + PP-OCRv6_medium_rec"
    )
    assert engine.model == "PP-OCRv6_medium"
