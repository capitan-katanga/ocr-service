from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

EngineName = Literal["paddleocr"]
Confidence = Annotated[float, Field(ge=0, le=1)]


class Camel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Line(Camel):
    text: str
    confidence: Confidence | None
    bbox: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=4, max_length=4)] | None


class Page(Camel):
    number: Annotated[int, Field(ge=1)]
    width: Annotated[int, Field(ge=1)]
    height: Annotated[int, Field(ge=1)]
    rotation: Literal[0, 90, 180, 270]
    lines: list[Line]


class OcrResult(Camel):
    engine: EngineName
    engine_version: str
    config_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    model: str
    duration_ms: Annotated[int, Field(ge=0)]
    mean_confidence: Confidence | None
    page_count: Annotated[int, Field(ge=1)]
    truncated: bool
    pages: list[Page]
    layout_text: str


class EngineInfo(Camel):
    name: EngineName
    enabled: bool
    ready: bool
    engine_version: str
    model: str
    config_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    provides_confidence: bool
    provides_bbox: bool


class EngineList(Camel):
    default_engine: EngineName
    engines: list[EngineInfo]


class HealthStatus(Camel):
    status: Literal["UP", "DOWN"]
    engines: dict[str, Literal["UP", "DOWN", "LOADING", "DISABLED"]] | None = None


class ProblemDetail(Camel):
    """Solo para documentar el OpenAPI; las respuestas de error se arman en `errors.problem_response`."""

    type: str
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    code: str
    retryable: bool
    engine: EngineName | None = None
