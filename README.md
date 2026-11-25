# Agro Scoring API v2 — ЛР №2

Сквозной `request_id`, middleware, SQLite (aiosqlite), explicit и dynamic batching, backpressure,
таймаут инференса, внешняя LLM (Vireonix) через `httpx.AsyncClient`.

## Запуск
```bash
pip install -r requirements.txt
python seed_db.py            # 120 тестовых прогнозов за 4 недели (для пагинации по дням недели)
uvicorn main:app --reload    # Swagger: http://127.0.0.1:8000/docs
pytest -v                    # 22 теста, LLM подменяется заглушкой
```

## Демонстрации
| Что | Как |
|---|---|
| 429 (переполнение очереди) | `$env:QUEUE_MAX_SIZE=5; $env:MODEL_LATENCY_SECONDS=0.5; uvicorn main:app`, в другом терминале `python load_test.py 40` |
| 504 (таймаут модели) | `$env:INFERENCE_TIMEOUT_SECONDS=0.001; uvicorn main:app`, затем POST /predict |
| 409 (batch request_id) | POST /batch-predict, затем GET /predictions/{request_id} |
| Ошибка LLM | `$env:LLM_BASE_URL="http://127.0.0.1:9/v1"; uvicorn main:app`, затем POST /llm/chat → 503 |

После демонстрации закройте терминал или выполните `Remove-Item Env:QUEUE_MAX_SIZE` и т. п.

## Структура
| Файл | Назначение |
|---|---|
| main.py | lifespan, middleware, endpoints |
| config.py | параметры (MAX_BATCH_SIZE, QUEUE_MAX_SIZE, таймауты, LLM) |
| schemas.py | Pydantic-схемы |
| model.py | model_predict / model_predict_batch |
| services.py | бизнес-правила, инференс с таймаутом, постпроцессинг |
| batcher.py | asyncio.Queue + воркер dynamic batching |
| database.py | SQLite: запись, пакетная запись, поиск, пагинация |
| llm_client.py | клиент LLM и перевод ошибок в 502/503/504 |
| errors.py | ошибки и обработчики (ответ всегда с request_id) |
| seed_db.py, load_test.py | тестовые данные, нагрузочная проверка |
