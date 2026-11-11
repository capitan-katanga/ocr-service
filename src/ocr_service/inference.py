import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from ocr_service.errors import ErrorCode, OcrError


class InferenceGate:
    """`concurrency` inferencias en curso y `max_queue` esperando; lo que sobra se rechaza sin esperar."""

    def __init__(self, concurrency: int, max_queue: int, timeout: float) -> None:
        self._capacity = concurrency + max_queue
        self._timeout = timeout
        self._used = 0
        self._running = 0
        self._sem = asyncio.Semaphore(concurrency)
        self._executor = ThreadPoolExecutor(concurrency, thread_name_prefix="ocr-infer")

    @property
    def queued(self) -> int:
        return self._used - self._running

    async def run[T](self, fn: Callable[[], T]) -> T:
        if self._used >= self._capacity:
            raise OcrError(ErrorCode.OVERLOADED, "Inference queue is full.")
        self._used += 1
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        try:
            await asyncio.wait_for(self._sem.acquire(), self._timeout)
        except BaseException as e:
            self._used -= 1
            if isinstance(e, TimeoutError):
                raise OcrError(ErrorCode.ENGINE_TIMEOUT, "Timed out waiting for inference.") from e
            raise
        self._running += 1
        released = False

        def release(_: object = None) -> None:
            nonlocal released
            if not released:
                released = True
                self._running -= 1
                self._used -= 1
                self._sem.release()

        fut = loop.run_in_executor(self._executor, fn)
        # el lugar se libera cuando el hilo termina, no cuando vence el timeout: Paddle no se puede cortar
        fut.add_done_callback(release)
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # sin "never retrieved" tras un timeout
        try:
            return await asyncio.wait_for(asyncio.shield(fut), max(0.0, deadline - loop.time()))
        except TimeoutError as e:
            raise OcrError(ErrorCode.ENGINE_TIMEOUT, "Inference timed out.") from e
        finally:
            if fut.done():
                release()  # sin esperar al callback: el próximo request ya ve el lugar libre
