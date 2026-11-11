import asyncio
import time
from typing import Annotated, Any

from fastapi import APIRouter, Form, Request, UploadFile

from ocr_service.schemas import OcrResult, ProblemDetail
from ocr_service.telemetry import EXTRACTION_PATH
from ocr_service.validation import MAX_BYTES

router = APIRouter()

PROBLEM: dict[int | str, dict[str, Any]] = {
    code: {
        "model": ProblemDetail,
        "content": {"application/problem+json": {"schema": {"$ref": "#/components/schemas/ProblemDetail"}}},
    }
    for code in (400, 413, 415, 422, 500, 503, 504)
}


@router.post(EXTRACTION_PATH, tags=["extractions"], operation_id="createExtraction", responses=PROBLEM)
async def create_extraction(
    request: Request, file: UploadFile, engine: Annotated[str | None, Form()] = None
) -> OcrResult:
    started = time.perf_counter()
    st = request.app.state
    tel = st.telemetry
    wanted = engine or st.registry.default
    if (model := st.registry.model_of(wanted)) is not None:  # para el problem+json y las métricas, también si falla
        request.state.engine, request.state.model = wanted, model
    name, ocr_engine = st.registry.resolve(engine)
    attrs = {"model.name": ocr_engine.model}

    data = await file.read(MAX_BYTES + 1)
    kind = st.validator.validate(data)

    with tel.tracer.start_as_current_span("ocr preprocess", attributes=attrs):
        prepared = await asyncio.to_thread(st.preprocessor.prepare, data, kind)
    with tel.tracer.start_as_current_span("ocr infer", attributes=attrs):
        # todas las páginas en un solo lugar de la cola y bajo un solo timeout
        results = await st.gate.run(lambda: [ocr_engine.recognize(p.image) for p in prepared.pages])
    with tel.tracer.start_as_current_span("ocr layout", attributes=attrs):
        page_lines = [r.lines for r in results]
        layout_text = st.layout.layout_text(page_lines)
        mean = st.layout.mean_confidence(page_lines)

    pages = [
        {
            "number": p.number,
            "width": r.width,
            "height": r.height,
            "rotation": (p.rotation + r.angle) % 360,
            "lines": [{"text": ln.text, "confidence": ln.confidence, "bbox": ln.bbox} for ln in r.lines],
        }
        for p, r in zip(prepared.pages, results, strict=True)
    ]
    if mean is not None:
        tel.confidence.record(mean, {"ocr.engine": name, "model.name": ocr_engine.model})
    return OcrResult.model_validate(
        {
            "engine": name,
            "engine_version": ocr_engine.engine_version,
            "config_hash": st.registry.config_hash(name),
            "model": ocr_engine.model,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "mean_confidence": mean,
            "page_count": prepared.page_count,
            "truncated": prepared.truncated,
            "pages": pages,
            "layout_text": layout_text,
        }
    )
