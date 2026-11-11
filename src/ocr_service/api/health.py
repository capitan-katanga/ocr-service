from fastapi import APIRouter, Request, Response

from ocr_service.schemas import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health/live", operation_id="liveness", response_model_exclude_none=True)
async def live() -> HealthStatus:
    return HealthStatus(status="UP")


@router.get(
    "/health/ready",
    operation_id="readiness",
    responses={503: {"model": HealthStatus}},
    response_model_exclude_none=True,
)
async def ready(request: Request, response: Response) -> HealthStatus:
    registry = request.app.state.registry
    if not registry.ready:
        response.status_code = 503
    return HealthStatus(status="UP" if registry.ready else "DOWN", engines=registry.state)
