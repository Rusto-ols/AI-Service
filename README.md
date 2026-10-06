# Agro Scoring API — ЛР №1 «Проектирование AI-сервиса на FastAPI»

## Запуск
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload
```
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI:    http://127.0.0.1:8000/openapi.json

## Тесты (таблица тест-кейсов, Задание 10)
```bash
pytest -v
```

## Структура
| Файл | Назначение |
|---|---|
| main.py | API-слой: маршруты, HTTP-коды, описание endpoints, логирование |
| schemas.py | Pydantic-схемы FarmRequest, PredictionResponse, ModelInfo |
| model.py | calculate_risk() и метаданные модели |
| services.py | бизнес-валидация региона, постпроцессинг, рекомендации |
| storage.py | хранилище прогнозов (dict в памяти) |
| tests/test_api.py | 10 тест-кейсов |

## Тестовый запрос для /predict
```json
{"farm_id":"FARM-001","region":"Krasnodar","crop_type":"wheat","area_ha":2500,
 "temperature_avg":24.3,"precipitation_mm":320,"payment_delay_days":45,
 "previous_defaults":1,"debt":6500000}
```
