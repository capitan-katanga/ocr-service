"""Stage `models` del Dockerfile: baja los modelos a OCR_MODELS_DIR/<model_name>/ (sin descargas en runtime).

Prende los tres flags para que el pipeline cargue (y por lo tanto baje) también el clasificador de orientación de
documento, UVDoc y la orientación de línea: prenderlos después es configuración, no una descarga.
"""

import os
import shutil
from pathlib import Path

from paddleocr import PaddleOCR

from ocr_service.engines.paddle import DOC_ORIENTATION_MODEL, DOC_UNWARPING_MODEL, TEXTLINE_ORIENTATION_MODEL
from ocr_service.settings import Settings

s = Settings()
models = [
    s.ocr_text_detection_model,
    s.ocr_text_recognition_model,
    DOC_ORIENTATION_MODEL,
    DOC_UNWARPING_MODEL,
    TEXTLINE_ORIENTATION_MODEL,
]

PaddleOCR(
    engine="paddle",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True,
    text_detection_model_name=s.ocr_text_detection_model,
    text_recognition_model_name=s.ocr_text_recognition_model,
    doc_orientation_classify_model_name=DOC_ORIENTATION_MODEL,
    doc_unwarping_model_name=DOC_UNWARPING_MODEL,
    textline_orientation_model_name=TEXTLINE_ORIENTATION_MODEL,
)

cache = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex")) / "official_models"
for name in models:
    shutil.copytree(cache / name, Path(s.ocr_models_dir) / name, dirs_exist_ok=True)
