from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Env vars sin prefijo (pydantic-settings es case-insensitive): OCR_ENGINE, OCR_MAX_QUEUE, OTEL_*..."""

    ocr_engine: str = "paddleocr"
    ocr_enabled_engines: str = "paddleocr"  # lista separada por comas
    ocr_max_concurrency: int = 1
    ocr_max_queue: int = 4
    ocr_inference_timeout: float = 50
    ocr_cpu_threads: int = 4
    ocr_pdf_dpi: int = 200
    ocr_max_pdf_pages: int = 5
    ocr_max_image_pixels: int = 40_000_000
    ocr_min_image_side: int = 32
    ocr_use_doc_orientation_classify: bool = False
    ocr_use_doc_unwarping: bool = False
    ocr_use_textline_orientation: bool = False
    ocr_text_detection_model: str = "PP-OCRv6_medium_det"
    ocr_text_recognition_model: str = "PP-OCRv6_medium_rec"
    ocr_models_dir: str = "/models"
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "ocr-service"

    @property
    def enabled_engines(self) -> set[str]:
        return {e.strip() for e in self.ocr_enabled_engines.split(",") if e.strip()}
