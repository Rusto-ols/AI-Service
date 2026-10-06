
_predictions: dict[str, dict] = {}


def save(result: dict) -> None:
    _predictions[result["request_id"]] = result


def get(request_id: str) -> dict | None:
    return _predictions.get(request_id)


def list_latest(limit: int, risk_level: str | None = None) -> list[dict]:
    values = list(reversed(_predictions.values()))  # dict хранит порядок вставки
    if risk_level is not None:
        values = [item for item in values if item["risk_level"] == risk_level]
    return values[:limit]


def clear() -> None:
    _predictions.clear()
