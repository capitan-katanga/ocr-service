import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from opentelemetry import baggage, context, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ocr_service.errors import ErrorCode, problem_response
from ocr_service.settings import Settings

log = logging.getLogger("ocr_service.http")

EXTRACTION_PATH = "/api/v1/ocr/extractions"
DURATION_BUCKETS = [0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie", "x-api-key", "proxy-authorization"}


def outcome_of(status: int) -> str:
    if status < 400:
        return "success"
    if status < 500:
        return "rejected"
    return "timeout" if status == 504 else "unavailable"


class Telemetry:
    def __init__(
        self, tracer_provider: TracerProvider, meter_provider: MeterProvider, logger_provider: LoggerProvider
    ) -> None:
        self.tracer_provider = tracer_provider
        self.meter_provider = meter_provider
        self.logger_provider = logger_provider
        self.tracer = tracer_provider.get_tracer("ocr_service")
        meter = meter_provider.get_meter("ocr_service")
        self._meter = meter
        self.duration = meter.create_histogram(
            "ocr.extraction.duration", unit="s", explicit_bucket_boundaries_advisory=DURATION_BUCKETS
        )
        self.confidence = meter.create_histogram("ocr.extraction.confidence")
        self.errors = meter.create_counter("ocr.extraction.errors")

    @classmethod
    def otlp(cls, settings: Settings) -> "Telemetry":
        base = settings.otel_exporter_otlp_endpoint.rstrip("/")
        resource = Resource.create({"service.name": settings.otel_service_name})
        tracer = TracerProvider(resource=resource)
        tracer.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces")))
        meter = MeterProvider(
            resource=resource,
            metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{base}/v1/metrics"))],
        )
        logger = LoggerProvider(resource=resource)
        logger.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{base}/v1/logs")))
        return cls(tracer, meter, logger)

    def observe_queue(self, depth: Callable[[], int]) -> None:
        def callback(_: CallbackOptions) -> list[Observation]:
            return [Observation(depth())]

        self._meter.create_observable_gauge("ocr.inference.queued", callbacks=[callback])

    def log_handler(self) -> logging.Handler:
        handler = LoggingHandler(level=logging.INFO, logger_provider=self.logger_provider)
        handler.addFilter(ContextFilter())
        return handler

    def shutdown(self) -> None:
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()
        self.logger_provider.shutdown()


class ContextFilter(logging.Filter):
    """Agrega traceId, spanId, correlationId y userId a cada log (los atributos del MDC en el lado Java)."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = trace.get_current_span().get_span_context()
        record.traceId = format(ctx.trace_id, "032x") if ctx.is_valid else ""
        record.spanId = format(ctx.span_id, "016x") if ctx.is_valid else ""
        record.correlationId = str(baggage.get_baggage("correlationId") or "")
        record.userId = str(baggage.get_baggage("userId") or "")
        return True


def install_telemetry(app: FastAPI, telemetry: Telemetry) -> None:
    app.state.telemetry = telemetry

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        started = time.perf_counter()
        ctx = trace.get_current_span().get_span_context()
        trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else None
        cid = baggage.get_baggage("correlationId") or request.headers.get("x-correlation-id") or trace_id
        token = context.attach(baggage.set_baggage("correlationId", str(cid))) if cid else None
        try:
            try:
                response = await call_next(request)
            except Exception:
                log.exception("Unhandled error")
                response = problem_response(request, ErrorCode.INTERNAL_ERROR)
            if cid:
                response.headers["X-Correlation-Id"] = str(cid)
            if trace_id:
                response.headers["X-Trace-Id"] = trace_id
            elapsed = time.perf_counter() - started
            if request.url.path == EXTRACTION_PATH:
                _record_duration(request, response.status_code, elapsed)
            _log_request(request, response, elapsed)
            return response
        finally:
            if token:
                context.detach(token)


def _record_duration(request: Request, status: int, elapsed: float) -> None:
    registry = request.app.state.registry
    engine = getattr(request.state, "engine", None) or registry.default  # rechazado antes de elegir motor
    attrs = {"ocr.engine": engine, "model.name": registry.model_of(engine) or "", "outcome": outcome_of(status)}
    request.app.state.telemetry.duration.record(elapsed, attrs)


def _log_request(request: Request, response: Response, elapsed: float) -> None:
    headers = {k: ("***" if k.lower() in SENSITIVE_HEADERS else v) for k, v in request.headers.items()}
    log.info(
        "%s %s -> %d in %dms (request %s bytes, response %s bytes) headers=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed * 1000,
        request.headers.get("content-length", "?"),
        response.headers.get("content-length", "?"),
        headers,
    )


def instrument(app: FastAPI, telemetry: Telemetry) -> None:
    # va después de install_telemetry: el span de servidor tiene que envolver al middleware
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=telemetry.tracer_provider,
        meter_provider=telemetry.meter_provider,
        excluded_urls="/health/live,/health/ready",
    )
