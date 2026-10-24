import uuid

from model import MODEL_VERSION, calculate_risk
from schemas import FarmRequest

ALLOWED_REGIONS = {"Krasnodar", "Rostov", "Stavropol"}


class UnknownRegionError(ValueError):
    """Регион прошёл проверку типа, но не обслуживается сервисом."""


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


def validate_business_rules(request: FarmRequest) -> None:
    if request.region not in ALLOWED_REGIONS:
        raise UnknownRegionError(
            f"Unknown region '{request.region}'. Allowed: {sorted(ALLOWED_REGIONS)}"
        )


def make_prediction(request: FarmRequest) -> dict:
    validate_business_rules(request)
    score = calculate_risk(request)
    level = risk_level(score)
    return {
        "request_id": str(uuid.uuid4()),
        "farm_id": request.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation(level),
        "model_version": MODEL_VERSION,
    }
