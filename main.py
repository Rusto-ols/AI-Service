"""API-слой Agro Scoring API v2: request_id, middleware, SQLite, batching, очередь, LLM."""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from math import ceil
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, Query, Request

import config
import database
from batcher import DynamicBatcher
from errors import (AmbiguousResultError, BusinessRuleError, NotFoundError, register_error_handlers)
from llm_client import LLMClient
from model import MODEL_NAME, MODEL_TYPE, MODEL_VERSION
from schemas import (BatchPredictionResponse, BatchRequest, ErrorResponse, FarmRequest,
                     LLMChatRequest, LLMChatResponse, PaginatedPredictions, PredictionResponse,
                     PredictWithLLMResponse, QueueStatus, RiskLevel)
from services import predict_many, predict_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("agro_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Старт: БД, воркер очереди, HTTP-клиент LLM. Остановка: всё закрываем."""
    await database.init_db()
    app.state.batcher = DynamicBatcher()
    app.state.batcher.start()
    http_client = httpx.AsyncClient(base_url=config.LLM_BASE_URL,
                                    timeout=config.LLM_TIMEOUT_SECONDS,
                                    headers={"Content-Type": "application/json"})
    app.state.llm = LLMClient(http_client)
    logger.info("startup: db=%s llm=%s queue_max=%s", config.DB_PATH, config.LLM_BASE_URL,
                config.QUEUE_MAX_SIZE)
    yield
    await app.state.batcher.stop()
    await http_client.aclose()


app = FastAPI(
    title="Agro Scoring API",
    description="API сервиса агроскоринга v2: сквозной request_id, SQLite, пакетный и "
                "очередной инференс, объяснение результата с помощью LLM.",
    version="2.0.0",
    lifespan=lifespan,
)
register_error_handlers(app)

ERRORS = {code: {"model": ErrorResponse} for code in (400, 404, 409, 422, 429, 500, 502, 503, 504)}


def errors(*codes: int) -> dict:
    return {c: ERRORS[c] for c in codes}


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """request_id из X-Request-ID или новый UUID, время обработки, заголовки, лог."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.6f}"
    logger.info("request_id=%s %s %s -> %s %.1f ms", request_id, request.method,
                request.url.path, response.status_code, process_time * 1000)
    return response


def rid(request: Request) -> str:
    return request.state.request_id


@app.get("/health", tags=["service"], summary="Проверка работоспособности")
async def health():
    return {"status": "ok"}


@app.get("/model-info", tags=["service"], summary="Информация о модели")
async def model_info():
    return {"model_name": MODEL_NAME, "model_version": MODEL_VERSION,
            "model_type": MODEL_TYPE, "status": "ready"}


@app.get("/queue-status", response_model=QueueStatus, tags=["service"],
         summary="Состояние очереди dynamic batching")
async def queue_status(request: Request):
    return {"queue_size": request.app.state.batcher.queue.qsize(),
            "queue_max_size": config.QUEUE_MAX_SIZE,
            "max_batch_size": config.MAX_BATCH_SIZE,
            "max_batch_wait_ms": config.MAX_BATCH_WAIT_MS,
            "inference_timeout_seconds": config.INFERENCE_TIMEOUT_SECONDS}


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"],
          summary="Оценка риска одного хозяйства",
          description="Инференс с таймаутом, затем запись в SQLite. 200 — только после commit.",
          responses=errors(400, 422, 503, 504))
async def predict(item: FarmRequest, request: Request):
    result = await predict_one(rid(request), item, source="single")
    result = await database.save_prediction(result)
    logger.info("request_id=%s prediction farm=%s score=%s level=%s", rid(request),
                result["farm_id"], result["risk_score"], result["risk_level"])
    return result


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["scoring"],
          summary="Явный пакетный инференс (explicit batching)",
          description=f"До {config.MAX_BATCH_SIZE} хозяйств: один вызов model_predict_batch "
                      "и одна транзакция executemany + commit.",
          responses=errors(400, 422, 503, 504))
async def batch_predict(body: BatchRequest, request: Request):
    results = await predict_many(rid(request), body.items, source="batch")
    results = await database.save_predictions_batch(results)
    logger.info("request_id=%s batch size=%s", rid(request), len(results))
    return {"request_id": rid(request), "count": len(results), "items": results}


@app.post("/predict-queued", response_model=PredictionResponse, tags=["scoring"],
          summary="Инференс через очередь (dynamic batching)",
          description=f"Запрос попадает в asyncio.Queue, воркер собирает пакет до "
                      f"{config.MAX_BATCH_SIZE} шт. или {config.MAX_BATCH_WAIT_MS} мс. "
                      "При переполнении очереди — 429.",
          responses=errors(400, 422, 429, 503, 504))
async def predict_queued(item: FarmRequest, request: Request):
    return await request.app.state.batcher.submit(rid(request), item)


@app.get("/predictions", response_model=list[PredictionResponse], tags=["history"],
         summary="История прогнозов из SQLite", responses=errors(422, 503))
async def get_predictions(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    risk_level: RiskLevel | None = Query(default=None),
):
    return await database.list_predictions(limit, offset, risk_level)


@app.get("/predictions/by-weekday", response_model=PaginatedPredictions, tags=["history"],
         summary="Пагинация прогнозов по дню недели (LIMIT/OFFSET + COUNT(*))",
         responses=errors(400, 404, 422, 503))
async def predictions_by_weekday(
    weekday: str = Query(examples=["monday"], description="monday ... sunday"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=5, ge=1, le=100),
):
    key = weekday.strip().lower()
    if key not in database.WEEKDAYS:
        raise BusinessRuleError(f"Unknown weekday: {weekday}. Allowed: {', '.join(database.WEEKDAYS)}")
    data, total = await database.page_by_weekday(database.WEEKDAYS[key], page, per_page)
    total_pages = ceil(total / per_page) if total else 0
    if total and page > total_pages:
        raise NotFoundError(f"Page {page} does not exist, total_pages={total_pages}")

    def link(p: int) -> str:
        return "/predictions/by-weekday?" + urlencode({"weekday": key, "page": p, "per_page": per_page})

    return {"data": data, "meta": {
        "weekday": key, "current_page": page, "per_page": per_page,
        "total_pages": total_pages, "total_items": total,
        "next_page": link(page + 1) if page < total_pages else None,
        "prev_page": link(page - 1) if page > 1 and total else None,
    }}


@app.get("/predictions/{request_id}", response_model=PredictionResponse, tags=["history"],
         summary="Прогноз по request_id", responses=errors(404, 409, 503))
async def get_prediction(request_id: str):
    rows = await database.get_by_request_id(request_id)
    if not rows:
        raise NotFoundError()
    if len(rows) > 1:
        raise AmbiguousResultError()
    return rows[0]


@app.get("/requests/{request_id}/predictions", response_model=list[PredictionResponse],
         tags=["history"], summary="Все прогнозы одного запроса (для batch)",
         responses=errors(404, 503))
async def get_request_predictions(request_id: str):
    rows = await database.get_by_request_id(request_id)
    if not rows:
        raise NotFoundError()
    return rows


SYSTEM_PROMPT = (
    "Ты помощник кредитного аналитика банка в сфере агроскоринга. Отвечай по-русски, кратко "
    "и по делу.")

EXPLAIN_PROMPT = (
    "Объясни оператору банка результат агроскоринга в 3–4 предложениях. Не меняй оценку и "
    "категорию риска, не придумывай новых цифр — только объясни, какие факторы на неё повлияли.\n"
    "Данные хозяйства: {farm}\n"
    "Результат модели: risk_score={score}, risk_level={level}, рекомендация: {rec}")


@app.post("/llm/chat", response_model=LLMChatResponse, tags=["llm"],
          summary="Прямой запрос к внешней LLM", responses=errors(422, 502, 503, 504))
async def llm_chat(body: LLMChatRequest, request: Request):
    result = await request.app.state.llm.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body.prompt}],
        model=body.model, temperature=body.temperature, max_tokens=body.max_tokens)
    logger.info("request_id=%s llm model=%s latency=%s ms", rid(request),
                result["returned_model"], result["latency_ms"])
    return {"request_id": rid(request), "requested_model": body.model, **result}


@app.post("/predict-with-llm", response_model=PredictWithLLMResponse, tags=["llm"],
          summary="Скоринг + текстовое объяснение от LLM",
          description="Оценку считает модель; LLM получает готовый результат и только объясняет его.",
          responses=errors(400, 422, 502, 503, 504))
async def predict_with_llm(item: FarmRequest, request: Request):
    result = await predict_one(rid(request), item, source="llm")
    result = await database.save_prediction(result)
    prompt = EXPLAIN_PROMPT.format(farm=item.model_dump_json(), score=result["risk_score"],
                                   level=result["risk_level"], rec=result["recommendation"])
    llm = await request.app.state.llm.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=400)
    return {"prediction": result, "explanation": llm["answer"],
            "llm_model": llm["returned_model"], "llm_latency_ms": llm["latency_ms"]}
