# ocr-service

Transcribe imágenes (PNG, JPEG) y PDF sin texto confiable con PaddleOCR (PP-OCRv6, CPU) y devuelve las líneas con
confianza y bbox más un `layoutText` listo para un LLM. No clasifica ni interpreta. Lo llama receipt-parser por HTTP;
no usa RabbitMQ, base de datos ni servicios externos. Contrato: `contracts/openapi/ocr-service-api.yaml`.

- `POST /api/v1/ocr/extractions` (multipart `file`, `engine` opcional)
- `GET /api/v1/engines`
- `GET /health/live`, `GET /health/ready` (503 hasta que cargan los modelos)

## Configuración

Variables de entorno, sin prefijo (defaults entre paréntesis).

| Variable | Default | Qué hace |
|---|---|---|
| `OCR_ENGINE` / `OCR_ENABLED_ENGINES` | `paddleocr` / `paddleocr` | Motor por defecto y habilitados (lista con comas) |
| `OCR_MAX_CONCURRENCY` / `OCR_MAX_QUEUE` | `1` / `4` | Inferencias en curso y en espera; lleno: 503 `OVERLOADED` |
| `OCR_INFERENCE_TIMEOUT` | `50` | Segundos; vencido: 504 `ENGINE_TIMEOUT` (el lugar sigue ocupado hasta que el hilo termina) |
| `OCR_CPU_THREADS` | `4` | Hilos de paddlepaddle |
| `OCR_PDF_DPI` / `OCR_MAX_PDF_PAGES` | `200` / `5` | Rasterizado de PDF y tope de páginas procesadas |
| `OCR_MAX_IMAGE_PIXELS` / `OCR_MIN_IMAGE_SIDE` | `40000000` / `32` | Fuera de rango: 422 `IMAGE_OUT_OF_BOUNDS` |
| `OCR_USE_DOC_ORIENTATION_CLASSIFY` / `OCR_USE_DOC_UNWARPING` / `OCR_USE_TEXTLINE_ORIENTATION` | `false` | Etapas opcionales de PaddleOCR (entran al `configHash`) |
| `OCR_TEXT_DETECTION_MODEL` / `OCR_TEXT_RECOGNITION_MODEL` | `PP-OCRv6_medium_det` / `PP-OCRv6_medium_rec` | Modelos |
| `OCR_MODELS_DIR` | `/models` | Directorio de modelos (`<dir>/<nombre del modelo>`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTLP/HTTP para traces, métricas y logs |
| `OTEL_SERVICE_NAME` | `ocr-service` | `service.name` |

Propaga `traceparent` y `baggage` (`correlationId`, `userId`); `X-Correlation-Id` queda de respaldo. Responde siempre
`X-Correlation-Id` y `X-Trace-Id`.

## Correr

Docker (la imagen incluye los modelos, ~2,7 GB; puerto 8084):

```bash
docker build -t ocr-service:local .
docker run --rm -p 127.0.0.1:8084:8084 ocr-service:local
```

En el compose de `tracker-infra` se levanta con el profile `ocr` (usa la imagen ya construida).

Local, sin Docker: los modelos van a `OCR_MODELS_DIR`; para generarlos (una vez, ~185 MB en `./models`, ignorado por git
y Docker), `OCR_MODELS_DIR=./models uv run python scripts/fetch_models.py` y después:

```bash
uv sync --frozen
OCR_MODELS_DIR=./models uv run uvicorn ocr_service.main:app --port 8084 --reload
```

Sin LGTM levantado, `OTEL_SDK_DISABLED=true` evita los errores de exportación. La app se crea al pedir `app`
(`main.__getattr__`), por eso se usa uvicorn y no `fastapi dev`.

## Tests

Comandos en `CLAUDE.md`. Los comprobantes de los tests son ficticios y se generan en el momento; no se versionan
comprobantes reales.
