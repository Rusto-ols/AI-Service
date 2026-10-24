"""Бизнес-логика: бизнес-валидация, инференс с таймаутом, постпроцессинг."""
import asyncio

import config
from errors import BusinessRuleError, InferenceTimeoutError
from model import MODEL_VERSION, model_predict, model_predict_batch
from schemas import FarmRequest

ALLOWED_REGIONS = {"Krasnodar", "Rostov", "Stavropol"}


def risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def recommendation(level: str) -> str:
    if level == "low":
        return "Риск низкий. Можно одобрить заявку в стандартном порядке."
    if level == "medium":
        return "Риск средний. Рекомендуется запросить дополнительные документы и обеспечение."
    return "Риск высокий. Рекомендуется отказать или передать заявку на ручную проверку андеррайтеру."


def validate_business_rules(item: FarmRequest) -> None:
    if item.region not in ALLOWED_REGIONS:
        raise BusinessRuleError(f"Unknown region: {item.region}")


def build_result(request_id: str, item: FarmRequest, score: float, source: str) -> dict:
    level = risk_level(score)
    return {
        "request_id": request_id,
        "farm_id": item.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation(level),
        "model_version": MODEL_VERSION,
        "source": source,
    }


async def predict_one(request_id: str, item: FarmRequest, source: str = "single") -> dict:
    validate_business_rules(item)
    try:
        score = await asyncio.wait_for(model_predict(item), timeout=config.INFERENCE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise InferenceTimeoutError() from exc
    return build_result(request_id, item, score, source)


async def predict_many(request_id: str, items: list[FarmRequest], source: str = "batch") -> list[dict]:
    for item in items:
        validate_business_rules(item)
    try:
        scores = await asyncio.wait_for(model_predict_batch(items),
                                        timeout=config.INFERENCE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise InferenceTimeoutError() from exc
    return [build_result(request_id, item, score, source) for item, score in zip(items, scores)]
