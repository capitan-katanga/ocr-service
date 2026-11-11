import os
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

import numpy as np
from PIL import Image

from ocr_service.engines.types import OcrLine, PageOcr
from ocr_service.settings import Settings

# nombres de modelo de los tres flags (los baja scripts/fetch_models.py a OCR_MODELS_DIR/<nombre>)
DOC_ORIENTATION_MODEL = "PP-LCNet_x1_0_doc_ori"
DOC_UNWARPING_MODEL = "UVDoc"
TEXTLINE_ORIENTATION_MODEL = "PP-LCNet_x1_0_textline_ori"


class PaddleOcrEngine:
    """Único módulo que conoce a PaddleOCR."""

    provides_confidence = True
    provides_bbox = True

    def __init__(self, settings: Settings, factory: Callable[..., Any] | None = None) -> None:
        self._s = settings
        self._factory = factory
        self._ocr: Any = None
        self.model = settings.ocr_text_recognition_model.removesuffix("_rec")
        self.engine_version = (
            f"paddleocr {version('paddleocr')}; paddlepaddle {version('paddlepaddle')}; "
            f"{settings.ocr_text_detection_model} + {settings.ocr_text_recognition_model}"
        )

    def load(self) -> None:
        s = self._s
        factory = self._factory
        if factory is None:
            from paddleocr import PaddleOCR

            factory = PaddleOCR

        def model_dir(name: str) -> str:
            return os.path.join(s.ocr_models_dir, name)

        # sin `lang`: con nombres de modelo explícitos PaddleOCR lo ignora
        self._ocr = factory(
            engine="paddle",
            cpu_threads=s.ocr_cpu_threads,
            use_doc_orientation_classify=s.ocr_use_doc_orientation_classify,
            use_doc_unwarping=s.ocr_use_doc_unwarping,
            use_textline_orientation=s.ocr_use_textline_orientation,
            text_detection_model_name=s.ocr_text_detection_model,
            text_detection_model_dir=model_dir(s.ocr_text_detection_model),
            text_recognition_model_name=s.ocr_text_recognition_model,
            text_recognition_model_dir=model_dir(s.ocr_text_recognition_model),
            doc_orientation_classify_model_name=DOC_ORIENTATION_MODEL,
            doc_orientation_classify_model_dir=model_dir(DOC_ORIENTATION_MODEL),
            doc_unwarping_model_name=DOC_UNWARPING_MODEL,
            doc_unwarping_model_dir=model_dir(DOC_UNWARPING_MODEL),
            textline_orientation_model_name=TEXTLINE_ORIENTATION_MODEL,
            textline_orientation_model_dir=model_dir(TEXTLINE_ORIENTATION_MODEL),
        )

    def recognize(self, image: Image.Image) -> PageOcr:
        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]  # PaddleX espera BGR como cv2
        res = self._ocr.predict(np.ascontiguousarray(bgr))[0]
        pre = res["doc_preprocessor_res"]
        height, width = pre["output_img"].shape[:2]
        angle = int(pre.get("angle", 0) or 0)
        lines = [
            OcrLine(
                text=str(text),
                confidence=min(1.0, max(0.0, float(score))),
                bbox=(max(0, int(x0)), max(0, int(y0)), max(0, int(x1)), max(0, int(y1))),
            )
            for text, score, (x0, y0, x1, y1) in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"], strict=True)
        ]
        return PageOcr(lines=lines, width=int(width), height=int(height), angle=angle if angle in (90, 180, 270) else 0)
