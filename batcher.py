import asyncio
import logging
import time
from dataclasses import dataclass, field

import config
import database
from errors import AppError, InferenceTimeoutError, QueueFullError
from schemas import FarmRequest
from services import build_result, validate_business_rules
from model import model_predict_batch

logger = logging.getLogger("agro_api")


@dataclass
class QueuedItem:
    request_id: str
    data: FarmRequest
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.perf_counter)


class DynamicBatcher:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[QueuedItem] = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE)
        self._worker: asyncio.Task | None = None
        self.processed_batches = 0

    def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="batch-worker")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

    async def submit(self, request_id: str, data: FarmRequest) -> dict:
        validate_business_rules(data)
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(QueuedItem(request_id, data, future))
        except asyncio.QueueFull as exc:
            raise QueueFullError() from exc
        return await future

    async def _collect_batch(self) -> list[QueuedItem]:
        first = await self.queue.get()
        batch = [first]
        deadline = time.perf_counter() + config.MAX_BATCH_WAIT_MS / 1000
        while len(batch) < config.MAX_BATCH_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def _process(self, batch: list[QueuedItem]) -> None:
        try:
            scores = await asyncio.wait_for(model_predict_batch([i.data for i in batch]),
                                            timeout=config.INFERENCE_TIMEOUT_SECONDS)
            results = [build_result(i.request_id, i.data, s, "queued") for i, s in zip(batch, scores)]
            results = await database.save_predictions_batch(results)
        except asyncio.TimeoutError:
            error: Exception = InferenceTimeoutError()
        except AppError as exc:
            error = exc
        except Exception as exc:
            logger.exception("batch worker failed")
            error = exc
        else:
            self.processed_batches += 1
            logger.info("dynamic batch #%s size=%s request_ids=%s", self.processed_batches,
                        len(batch), [i.request_id for i in batch])
            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)
            return
        for item in batch:
            if not item.future.done():
                item.future.set_exception(error)

    async def _run(self) -> None:
        while True:
            batch = await self._collect_batch()
            await self._process(batch)
            for _ in batch:
                self.queue.task_done()
