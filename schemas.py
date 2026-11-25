"""Pydantic-схемы: контракт входных и выходных данных API."""
from typing import Literal

from pydantic import BaseModel, Field

import config

RiskLevel = Literal["low", "medium", "high"]


class FarmRequest(BaseModel):
    """Сведения о сельскохозяйственном предприятии для оценки риска."""

    farm_id: str = Field(min_length=1, examples=["FARM-001"])
    region: str = Field(examples=["Krasnodar"])
    crop_type: str = Field(examples=["wheat"])
    area_ha: float = Field(gt=0, description="Площадь посевов, га", examples=[2500])
    temperature_avg: float = Field(description="Средняя температура, °C", examples=[24.3])
    precipitation_mm: float = Field(ge=0, description="Осадки, мм", examples=[320])
    payment_delay_days: int = Field(ge=0, description="Просрочка платежа, дней", examples=[45])
    previous_defaults: int = Field(ge=0, description="Число прошлых дефолтов", examples=[1])
    debt: float = Field(ge=0, description="Текущая задолженность, руб.", examples=[6500000])


class PredictionResponse(BaseModel):
    request_id: str = Field(description="Сквозной идентификатор HTTP-запроса")
    farm_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    recommendation: str
    model_version: str
    source: Literal["single", "batch", "queued", "llm"] = Field(description="Режим инференса")
    created_at: str = Field(description="Время сохранения в БД (UTC, ISO 8601)")


class BatchRequest(BaseModel):
    items: list[FarmRequest] = Field(min_length=1, max_length=config.MAX_BATCH_SIZE,
                                     description=f"От 1 до {config.MAX_BATCH_SIZE} хозяйств")


class BatchPredictionResponse(BaseModel):
    request_id: str
    count: int
    items: list[PredictionResponse]


class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: str | None
    prev_page: str | None


class PaginatedPredictions(BaseModel):
    data: list[PredictionResponse]
    meta: PaginationMeta


class QueueStatus(BaseModel):
    queue_size: int
    queue_max_size: int
    max_batch_size: int
    max_batch_wait_ms: int
    inference_timeout_seconds: float


class LLMChatRequest(BaseModel):
    prompt: str = Field(min_length=1, examples=["Объясни, что такое агроскоринг"])
    model: str = Field(default=config.LLM_MODEL, examples=["auto"])
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=300, ge=1, le=4000)


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str | None
    answer: str
    finish_reason: str | None
    latency_ms: float


class PredictWithLLMResponse(BaseModel):
    prediction: PredictionResponse = Field(description="Результат скоринга (LLM его не меняет)")
    explanation: str = Field(description="Текстовое объяснение от LLM")
    llm_model: str | None
    llm_latency_ms: float


class ErrorResponse(BaseModel):
    detail: str | list
    request_id: str | None = None
