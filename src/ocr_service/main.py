import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from ocr_service.api import engines as engines_api
from ocr_service.api import extraction, health
from ocr_service.engines import EngineRegistry
from ocr_service.engines.paddle import PaddleOcrEngine
from ocr_service.engines.types import OcrEngine
from ocr_service.errors import install_error_handlers
from ocr_service.inference import InferenceGate
from ocr_service.layout import LayoutReconstructor
from ocr_service.preprocessing import Preprocessor
from ocr_service.settings import Settings
from ocr_service.telemetry import Telemetry, install_telemetry, instrument
from ocr_service.validation import InputValidator


def create_app(
    settings: Settings | None = None, engine: OcrEngine | None = None, telemetry: Telemetry | None = None
) -> FastAPI:
    settings = settings or Settings()
    owns_telemetry = telemetry is None
    telemetry = telemetry or Telemetry.otlp(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        handler = telemetry.log_handler()
        root = logging.getLogger()
        root.addHandler(handler)
        logging.getLogger("ocr_service").setLevel(logging.INFO)
        # los modelos cargan en un hilo: el servicio atiende /health y responde 503 ENGINE_NOT_READY mientras tanto
        loading = asyncio.create_task(asyncio.to_thread(app.state.registry.load_all))
        try:
            yield
        finally:
            await loading
            root.removeHandler(handler)
            if owns_telemetry:
                telemetry.shutdown()

    app = FastAPI(title="ocr-service API", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = EngineRegistry(settings, {"paddleocr": engine or PaddleOcrEngine(settings)})
    app.state.gate = InferenceGate(settings.ocr_max_concurrency, settings.ocr_max_queue, settings.ocr_inference_timeout)
    app.state.validator = InputValidator(settings)
    app.state.preprocessor = Preprocessor(settings)
    app.state.layout = LayoutReconstructor()
    telemetry.observe_queue(lambda: app.state.gate.queued)
    install_error_handlers(app)
    app.include_router(extraction.router)
    app.include_router(engines_api.router)
    app.include_router(health.router)
    install_telemetry(app, telemetry)
    instrument(app, telemetry)
    return app


def __getattr__(name: str) -> Any:
    # uvicorn pide `app`; crearlo recién acá evita levantar los exportadores OTLP al importar el módulo en los tests
    if name == "app":
        return create_app()
    raise AttributeError(name)
