import asyncio
import functools
import threading
from collections.abc import Awaitable, Callable

import pytest

from ocr_service.errors import ErrorCode, OcrError
from ocr_service.inference import InferenceGate


def sync(fn: Callable[[], Awaitable[None]]) -> Callable[[], None]:
    @functools.wraps(fn)
    def wrapper() -> None:
        asyncio.run(fn())

    return wrapper


def blocked(ev: threading.Event, value: int = 1) -> int:
    ev.wait(5)
    return value


@sync
async def test_runs_and_returns() -> None:
    gate = InferenceGate(1, 1, 5)
    assert await gate.run(lambda: 7) == 7


@sync
async def test_full_capacity_is_rejected_without_waiting() -> None:
    gate = InferenceGate(1, 1, 5)
    ev = threading.Event()
    running = asyncio.create_task(gate.run(lambda: blocked(ev)))
    queued = asyncio.create_task(gate.run(lambda: blocked(ev, 2)))
    await asyncio.sleep(0.1)
    assert gate.queued == 1
    with pytest.raises(OcrError) as e:
        await asyncio.wait_for(gate.run(lambda: 0), 0.5)
    assert e.value.code == ErrorCode.OVERLOADED
    ev.set()
    assert await asyncio.gather(running, queued) == [1, 2]


@sync
async def test_timeout_keeps_the_slot_until_the_thread_ends() -> None:
    gate = InferenceGate(1, 0, 0.1)
    ev = threading.Event()
    with pytest.raises(OcrError) as e:
        await gate.run(lambda: blocked(ev))
    assert e.value.code == ErrorCode.ENGINE_TIMEOUT
    with pytest.raises(OcrError) as e2:  # el hilo vencido sigue corriendo: no hay lugar
        await gate.run(lambda: 0)
    assert e2.value.code == ErrorCode.OVERLOADED
    ev.set()
    await asyncio.sleep(0.1)
    assert await gate.run(lambda: 3) == 3


@sync
async def test_timeout_while_waiting_in_queue_frees_the_queue_place() -> None:
    gate = InferenceGate(1, 1, 0.2)
    ev = threading.Event()
    running = asyncio.create_task(gate.run(lambda: blocked(ev)))
    await asyncio.sleep(0.05)
    queued = asyncio.create_task(gate.run(lambda: 0))
    results = await asyncio.gather(running, queued, return_exceptions=True)
    assert [r.code for r in results if isinstance(r, OcrError)] == [ErrorCode.ENGINE_TIMEOUT] * 2
    assert gate.queued == 0  # el de la cola salió; el vencido sigue ocupando el hilo
    ev.set()


@sync
async def test_engine_exception_propagates_and_frees_the_slot() -> None:
    gate = InferenceGate(1, 0, 5)

    def boom() -> int:
        raise ValueError("x")

    with pytest.raises(ValueError, match="x"):
        await gate.run(boom)
    assert await gate.run(lambda: 1) == 1


@sync
async def test_cancelled_while_queued_frees_the_place() -> None:
    gate = InferenceGate(1, 1, 5)
    ev = threading.Event()
    first = asyncio.create_task(gate.run(lambda: blocked(ev)))
    second = asyncio.create_task(gate.run(lambda: 0))
    await asyncio.sleep(0.05)
    second.cancel()
    await asyncio.sleep(0.05)
    assert gate.queued == 0
    ev.set()
    await first
