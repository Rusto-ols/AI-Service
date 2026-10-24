import logging
import time

from fastapi import FastAPI, HTTPException, Query, Request, status

import storage
from model import MODEL_NAME, MODEL_TYPE, MODEL_VERSION
from schemas import FarmRequest, HealthResponse, ModelInfo, PredictionResponse, RiskLevel
from services import UnknownRegionError, make_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("agro_api")

app = FastAPI(
    title="Agro Scoring API",
    description="API сервиса агроскоринга: оценка кредитного риска сельскохозяйственных предприятий.",
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path,
                response.status_code, elapsed_ms)
    return response


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["service"],
    summary="Проверка работоспособности",
    description="Проверяет, что API-процесс запущен и отвечает на запросы. Не обращается к модели.",
)
def health():
    return {"status": "ok"}


@app.get(
    "/model-info",
    response_model=ModelInfo,
    tags=["service"],
    summary="Информация о модели",
    description="Возвращает имя, версию, тип и состояние модели, которая выполняет инференс.",
)
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "status": "ready",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["scoring"],
    summary="Оценка риска хозяйства",
    description=(
        "Принимает сведения о хозяйстве, проверяет их (структурная и бизнес-валидация), "
        "рассчитывает risk_score, присваивает категорию риска и рекомендацию. "
        "Результат сохраняется и доступен по request_id."
    ),
    responses={400: {"description": "Регион не обслуживается"},
               422: {"description": "Ошибка валидации входных данных"}},
)
def predict(request: FarmRequest):
    try:
        result = make_prediction(request)
    except UnknownRegionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown region") from exc
    storage.save(result)
    logger.info("prediction %s farm=%s score=%s level=%s model=%s", result["request_id"],
                result["farm_id"], result["risk_score"], result["risk_level"], result["model_version"])
    return result


@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    tags=["scoring"],
    summary="Получение прогноза по ID",
    description="Возвращает сохранённый результат прогноза по его request_id.",
    responses={404: {"description": "Прогноз не найден"}},
)
def get_prediction(request_id: str):
    result = storage.get(request_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return result


@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
    tags=["scoring"],
    summary="Список последних прогнозов",
    description="Возвращает последние прогнозы (новые первыми) с ограничением limit и фильтром risk_level.",
)
def get_predictions(
    limit: int = Query(default=10, ge=1, le=100, description="Сколько записей вернуть (1–100)"),
    risk_level: RiskLevel | None = Query(default=None, description="Фильтр по категории риска"),
):
    return storage.list_latest(limit=limit, risk_level=risk_level)
