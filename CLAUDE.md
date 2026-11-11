# ocr-service

Servicio Python (FastAPI + PaddleOCR) que transcribe comprobantes (PNG, JPEG, PDF) para receipt-parser. Implementa
`contracts/openapi/ocr-service-api.yaml` del harness (`../../contracts/`), con copia en este repo. Decisión: ADR-019 en
`docs/architecture/architecture.json`; plan: `specs/FEAT-008-plan.md` (ambos del harness). Variables y uso: `README.md`.

## Verificación

```bash
uv sync --frozen && uv run ruff check . && uv run ruff format --check . && uv run mypy \
  && uv run pytest -q --cov --cov-report=xml --cov-fail-under=90
```

- CI (`.github/workflows/ci.yml`, actions fijadas por SHA): job `checks` con Trivy sobre `uv.lock` más este comando;
  job `image` con el build, readiness con `--network none` (sin intentos de descarga en los logs), `pytest -m slow` y
  Trivy sobre la imagen. Ambos Trivy cortan con HIGH/CRITICAL que tengan fix. Por eso la etapa `runtime` aplica
  `apt-get upgrade` y desinstala el pip del sistema (vendoriza urllib3, msgpack y setuptools viejos).
- El umbral de 90 % va en el comando y no en `addopts`: con él ahí, `pytest -m slow` fallaría por cobertura.
- Marker `slow` (excluido por defecto con `-m 'not slow'`): `tests/test_image.py` corre la imagen real con
  testcontainers. Requiere Docker y la imagen: `docker build -t ocr-service:local .` y `uv run pytest -m slow`.
- `contracts/openapi/ocr-service-api.yaml` es una copia del contrato del harness (que no tiene remoto): la leen
  Schemathesis y los tests de contrato. El contrato se cambia en el harness y se copia acá;
  `test_contract_copy_matches_harness` falla si difieren (se saltea fuera del harness, como en CI).
- mypy `--strict` sobre `src/ocr_service`; `ignore_missing_imports` solo para `paddleocr.*`, `paddlex.*`, `paddle.*`
  y `pypdfium2.*` (sin stubs ni `py.typed`).

## Convenciones

- Solo `engines/paddle.py` importa PaddleOCR (y lo hace dentro de `load()`); el resto del código y los tests usan
  un motor falso en ese límite.
- `paddleocr==3.7.0` y `paddlepaddle==3.2.0` (índice `paddle-cpu` en `pyproject.toml`) van con `==`. Subirlos exige
  regenerar la imagen y pasar el `slow`; la 3.x cambia nombres de modelo y API entre menores.
- Cambiar la salida de `LayoutReconstructor` exige subir `LAYOUT_VERSION` (entra al `configHash`).
- La imagen no descarga nada en runtime: `scripts/fetch_models.py` deja los modelos en `/models/<nombre>` y
  `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` apaga el chequeo de fuente de PaddleX.
- Errores siempre `application/problem+json` con `code` y `retryable` (`errors.ERROR_TABLE` es la tabla del contrato).
- Nada de comprobantes reales en el repo (tests, fixtures ni `output/`): los tests generan archivos ficticios.
