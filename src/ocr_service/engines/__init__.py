import hashlib
import json
import logging
from typing import Literal

from ocr_service import layout
from ocr_service.engines.types import OcrEngine
from ocr_service.errors import ErrorCode, OcrError
from ocr_service.settings import Settings

log = logging.getLogger(__name__)

State = Literal["UP", "DOWN", "LOADING", "DISABLED"]


class EngineRegistry:
    def __init__(self, settings: Settings, engines: dict[str, OcrEngine]) -> None:
        self._s = settings
        self._engines = engines
        self.state: dict[str, State] = {
            n: ("LOADING" if n in settings.enabled_engines else "DISABLED") for n in engines
        }

    @property
    def default(self) -> str:
        return self._s.ocr_engine

    def model_of(self, name: str) -> str | None:
        return self._engines[name].model if name in self._engines else None

    def load_all(self) -> None:
        for name, engine in self._engines.items():
            if self.state[name] == "DISABLED":
                continue
            try:
                engine.load()
                self.state[name] = "UP"
            except Exception:
                log.exception("Engine %s failed to load", name)
                self.state[name] = "DOWN"

    @property
    def ready(self) -> bool:
        return all(st in ("UP", "DISABLED") for st in self.state.values())

    def resolve(self, name: str | None) -> tuple[str, OcrEngine]:
        name = name or self.default
        if name not in self._engines:
            raise OcrError(ErrorCode.UNKNOWN_ENGINE, f"Engine {name} is not one of: {', '.join(self._engines)}.")
        if self.state[name] == "DISABLED":
            raise OcrError(ErrorCode.ENGINE_DISABLED, f"Engine {name} is not enabled.")
        if self.state[name] != "UP":
            raise OcrError(ErrorCode.ENGINE_NOT_READY, f"Engine {name} is {self.state[name].lower()}.")
        return name, self._engines[name]

    def config_hash(self, name: str) -> str:
        s = self._s
        models = {"det": s.ocr_text_detection_model, "rec": s.ocr_text_recognition_model}
        if s.ocr_use_textline_orientation:
            models["textline"] = "PP-LCNet_x1_0_textline_ori"
        config = {
            "engine": name,
            "models": models,
            "use_doc_orientation_classify": s.ocr_use_doc_orientation_classify,
            "use_doc_unwarping": s.ocr_use_doc_unwarping,
            "use_textline_orientation": s.ocr_use_textline_orientation,
            "pdf_dpi": s.ocr_pdf_dpi,
            "max_pdf_pages": s.ocr_max_pdf_pages,
            "layout_version": layout.LAYOUT_VERSION,
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def info(self) -> list[dict[str, object]]:
        return [
            {
                "name": n,
                "enabled": self.state[n] != "DISABLED",
                "ready": self.state[n] == "UP",
                "engine_version": e.engine_version,
                "model": e.model,
                "config_hash": self.config_hash(n),
                "provides_confidence": e.provides_confidence,
                "provides_bbox": e.provides_bbox,
            }
            for n, e in self._engines.items()
        ]
