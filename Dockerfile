FROM ghcr.io/astral-sh/uv:0.13.7-python3.13-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable

# libgomp1: OpenMP de paddle; libgl1 y libglib2.0-0: opencv-contrib (dependencia de paddlex)
FROM python:3.13-slim AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS models
COPY --from=builder /app/.venv /app/.venv
COPY scripts/fetch_models.py /tmp/fetch_models.py
ENV PATH="/app/.venv/bin:$PATH" OCR_MODELS_DIR=/models
RUN python /tmp/fetch_models.py

FROM base AS runtime
RUN useradd --system --create-home --uid 10001 ocr
COPY --from=builder /app/.venv /app/.venv
COPY --from=models /models /models
ENV PATH="/app/.venv/bin:$PATH" \
    OCR_MODELS_DIR=/models \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PYTHONUNBUFFERED=1
USER ocr
EXPOSE 8084
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=6 \
    CMD ["python", "-m", "ocr_service.healthcheck"]
CMD ["uvicorn", "ocr_service.main:app", "--host", "0.0.0.0", "--port", "8084", "--workers", "1"]
