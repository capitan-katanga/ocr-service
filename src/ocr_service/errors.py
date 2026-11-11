import logging
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    UNKNOWN_ENGINE = "UNKNOWN_ENGINE"
    ENGINE_DISABLED = "ENGINE_DISABLED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    CORRUPT_FILE = "CORRUPT_FILE"
    ENCRYPTED_PDF = "ENCRYPTED_PDF"
    IMAGE_OUT_OF_BOUNDS = "IMAGE_OUT_OF_BOUNDS"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    ENGINE_NOT_READY = "ENGINE_NOT_READY"
    OVERLOADED = "OVERLOADED"
    ENGINE_TIMEOUT = "ENGINE_TIMEOUT"


# código -> (status, retryable), igual que `ErrorCode` del contrato
ERROR_TABLE: dict[ErrorCode, tuple[int, bool]] = {
    ErrorCode.INVALID_REQUEST: (400, False),
    ErrorCode.UNKNOWN_ENGINE: (400, False),
    ErrorCode.ENGINE_DISABLED: (400, False),
    ErrorCode.PAYLOAD_TOO_LARGE: (413, False),
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: (415, False),
    ErrorCode.CORRUPT_FILE: (422, False),
    ErrorCode.ENCRYPTED_PDF: (422, False),
    ErrorCode.IMAGE_OUT_OF_BOUNDS: (422, False),
    ErrorCode.INTERNAL_ERROR: (500, True),
    ErrorCode.ENGINE_NOT_READY: (503, True),
    ErrorCode.OVERLOADED: (503, True),
    ErrorCode.ENGINE_TIMEOUT: (504, True),
}


class OcrError(Exception):
    def __init__(self, code: ErrorCode, detail: str | None = None) -> None:
        super().__init__(detail or code.value)
        self.code = code
        self.detail = detail


def problem_response(request: Request, code: ErrorCode, detail: str | None = None) -> JSONResponse:
    status, retryable = ERROR_TABLE[code]
    slug = code.value.lower().replace("_", "-")
    body: dict[str, Any] = {
        "type": f"https://expense-tracker.local/problems/{slug}",
        "title": code.value.replace("_", " ").capitalize(),
        "status": status,
        "code": code.value,
        "retryable": retryable,
        "instance": request.url.path,
    }
    if detail:
        body["detail"] = detail
    engine = getattr(request.state, "engine", None)
    if engine:
        body["engine"] = engine
    headers = {"Retry-After": "5"} if status == 503 else None
    request.app.state.telemetry.errors.add(1, {"ocr.engine": engine or "", "code": code.value})
    return JSONResponse(body, status_code=status, headers=headers, media_type="application/problem+json")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(OcrError)
    async def _ocr_error(request: Request, exc: OcrError) -> JSONResponse:
        return problem_response(request, exc.code, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # solo ubicación y mensaje del error: el input puede ser el comprobante
        detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return problem_response(request, ErrorCode.INVALID_REQUEST, detail)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> Response:
        if exc.status_code == 400:  # multipart malformado: lo levanta Starlette antes de validar
            return problem_response(request, ErrorCode.INVALID_REQUEST, str(exc.detail))
        return await http_exception_handler(request, exc)
