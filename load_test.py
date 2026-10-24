"""Нагрузочная проверка dynamic batching и backpressure.
Сервер запустить с маленькой очередью и медленной моделью, например (PowerShell):
    $env:QUEUE_MAX_SIZE="5"; $env:MODEL_LATENCY_SECONDS="0.5"; uvicorn main:app
Затем в другом терминале:  python load_test.py 50"""
import asyncio
import collections
import sys

import httpx

BODY = {"farm_id": "FARM-LOAD", "region": "Krasnodar", "crop_type": "wheat", "area_ha": 2500,
        "temperature_avg": 24.3, "precipitation_mm": 320, "payment_delay_days": 45,
        "previous_defaults": 1, "debt": 6500000}


async def main(n: int, url: str) -> None:
    async with httpx.AsyncClient(base_url=url, timeout=30, trust_env=False) as client:
        responses = await asyncio.gather(*(client.post("/predict-queued", json=BODY) for _ in range(n)),
                                         return_exceptions=True)
        status_resp = await client.get("/queue-status")
    errors = [r for r in responses if isinstance(r, Exception)]
    if errors:
        print(f"Не удалось подключиться ({len(errors)} запросов): {errors[0]!r}. Сервер запущен?")
    responses = [r for r in responses if not isinstance(r, Exception)]
    try:
        status = status_resp.json()
    except ValueError:
        status = f"HTTP {status_resp.status_code}: {status_resp.text[:200]!r}"
    counts = collections.Counter(r.status_code for r in responses)
    print(f"Отправлено {n} одновременных запросов на /predict-queued")
    for code, count in sorted(counts.items()):
        print(f"  HTTP {code}: {count}")
    sample = next((r for r in responses if r.status_code == 429), None)
    if sample is not None:
        print("Пример ответа 429:", sample.json())
    print("queue-status после нагрузки:", status)


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 50,
                     sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"))
