from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jsonschema_rs
import pytest
import schemathesis
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from schemathesis.specs.openapi.checks import (
    content_type_conformance,
    response_schema_conformance,
    status_code_conformance,
)

from ocr_service.main import create_app
from ocr_service.settings import Settings

from .conftest import FakeEngine, Obs, jpeg_bytes, pdf_bytes, png_bytes, wait_until

# el contrato vive en el harness, dos niveles arriba de este repo (revisar cuando el repo tenga CI propio)
CONTRACT = Path(__file__).resolve().parents[1] / "../../contracts/openapi/ocr-service-api.yaml"
# bytes válidos por tipo: con relleno al azar nunca se llegaría al 200
for media_type, sample in (("image/png", png_bytes()), ("image/jpeg", jpeg_bytes()), ("application/pdf", pdf_bytes())):
    schemathesis.openapi.media_type(media_type, st.just(sample))
CHECKS = (status_code_conformance, content_type_conformance, response_schema_conformance)


@pytest.fixture(scope="module")
def app() -> Iterator[FastAPI]:
    app = create_app(Settings(), FakeEngine(), Obs().telemetry)
    with TestClient(app):
        wait_until(lambda: app.state.registry.ready)
        yield app


@pytest.fixture(scope="module")
def api_schema(app: FastAPI) -> Any:
    schema = schemathesis.openapi.from_path(CONTRACT)
    schema.app = app
    return schema


schema = schemathesis.pytest.from_fixture("api_schema")


@schema.parametrize()
@settings(max_examples=40, deadline=None, suppress_health_check=list(HealthCheck))
def test_app_conforms_to_contract(case: Any) -> None:
    case.call_and_validate(checks=CHECKS)


def contract() -> dict[str, Any]:
    return yaml.safe_load(CONTRACT.read_text())  # type: ignore[no-any-return]


def test_success_response_matches_contract_schema(app: FastAPI) -> None:
    # Schemathesis genera bytes al azar y nunca llega a un 200: este valida OcrResult y EngineList del contrato
    spec = contract()
    client = TestClient(app)
    ok = client.post("/api/v1/ocr/extractions", files={"file": ("a.png", png_bytes(), "image/png")})
    engines = client.get("/api/v1/engines")
    for name, resp in (("OcrResult", ok), ("EngineList", engines)):
        assert resp.status_code == 200
        jsonschema_rs.validate({"$ref": f"#/components/schemas/{name}", "components": spec["components"]}, resp.json())


def deref(spec: dict[str, Any], node: Any) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        target: Any = spec
        for part in node["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        node = target
    return node


def required_paths(spec: dict[str, Any], node: Any, prefix: str = "") -> set[str]:
    node = deref(spec, node)
    out: set[str] = set()
    for alt in node.get("oneOf", []) + node.get("anyOf", []):
        out |= required_paths(spec, alt, prefix)
    if "items" in node:
        out |= required_paths(spec, node["items"], prefix + "[]")
    if isinstance(node.get("additionalProperties"), dict):
        out |= required_paths(spec, node["additionalProperties"], prefix + "*.")
    out |= {prefix + name for name in node.get("required", [])}
    for name, prop in node.get("properties", {}).items():
        out |= required_paths(spec, prop, f"{prefix}{name}.")
    return out


def summarize(spec: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    ops: dict[tuple[str, str], dict[str, Any]] = {}
    for path, item in spec["paths"].items():
        for method, op in item.items():
            body = op.get("requestBody", {}).get("content", {}).get("multipart/form-data", {}).get("schema")
            responses = {}
            for status, response in op["responses"].items():
                content = deref(spec, response).get("content", {})
                json_schema = next((c["schema"] for t, c in content.items() if t.endswith("json")), None)
                responses[str(status)] = required_paths(spec, json_schema) if json_schema else set()
            ops[(path, method)] = {
                "request": required_paths(spec, body) if body else set(),
                "responses": responses,
            }
    return ops


def test_generated_openapi_matches_contract(app: FastAPI) -> None:
    expected = summarize(contract())
    actual = summarize(app.openapi())
    assert actual.keys() == expected.keys()
    for op, exp in expected.items():
        assert actual[op]["request"] == exp["request"], op
        assert actual[op]["responses"].keys() == exp["responses"].keys(), op
        for status, paths in exp["responses"].items():
            assert actual[op]["responses"][status] == paths, (op, status)
