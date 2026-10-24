from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]


class FarmRequest(BaseModel):

    farm_id: str = Field(min_length=1, description="Идентификатор хозяйства", examples=["FARM-001"])
    region: str = Field(description="Регион", examples=["Krasnodar"])
    crop_type: str = Field(description="Тип основной культуры", examples=["wheat"])
    area_ha: float = Field(gt=0, description="Площадь посевов, га", examples=[2500])
    temperature_avg: float = Field(description="Средняя температура сезона, °C", examples=[24.3])
    precipitation_mm: float = Field(ge=0, description="Сумма осадков, мм", examples=[320])
    payment_delay_days: int = Field(ge=0, description="Текущая просрочка платежа, дней", examples=[15])
    previous_defaults: int = Field(ge=0, description="Количество прошлых просрочек/дефолтов", examples=[0])
    debt: float = Field(ge=0, description="Текущая задолженность, руб.", examples=[1500000])


class PredictionResponse(BaseModel):

    request_id: str = Field(description="Идентификатор прогноза (UUID)")
    farm_id: str
    risk_score: float = Field(ge=0, le=1, description="Числовая оценка риска 0..1")
    risk_level: RiskLevel = Field(description="Категория риска")
    recommendation: str = Field(description="Рекомендация оператору банка")
    model_version: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str


class HealthResponse(BaseModel):
    status: str
