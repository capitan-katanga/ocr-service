from fastapi import APIRouter, Request

from ocr_service.schemas import EngineList

router = APIRouter()


@router.get("/api/v1/engines", tags=["engines"], operation_id="listEngines")
async def list_engines(request: Request) -> EngineList:
    registry = request.app.state.registry
    return EngineList.model_validate({"default_engine": registry.default, "engines": registry.info()})
