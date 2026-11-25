"""Модель оценки риска (правило вместо ML-модели) — одиночный и пакетный инференс."""
import asyncio

import config
from schemas import FarmRequest

MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"


def calculate_risk(data: FarmRequest) -> float:
    score = 0.1
    if data.payment_delay_days > 30:
        score += 0.3
    if data.previous_defaults > 0:
        score += 0.3
    if data.debt > 5_000_000:
        score += 0.2
    if data.precipitation_mm < 100:
        score += 0.1
    return round(min(score, 1.0), 2)


async def model_predict(data: FarmRequest) -> float:
    """Одиночный инференс. Задержка имитирует фиксированную стоимость вызова модели."""
    await asyncio.sleep(config.MODEL_LATENCY_SECONDS)
    return calculate_risk(data)


async def model_predict_batch(items: list[FarmRequest]) -> list[float]:
    """Пакетный инференс: фиксированная стоимость вызова платится один раз на весь пакет."""
    await asyncio.sleep(config.MODEL_LATENCY_SECONDS)
    return [calculate_risk(item) for item in items]
