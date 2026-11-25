"""Заполняет БД тестовыми прогнозами за последние 4 недели — для демонстрации пагинации по дням недели.
Запуск: python seed_db.py [количество]   (по умолчанию 120)"""
import asyncio
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

import database
from schemas import FarmRequest
from services import ALLOWED_REGIONS, build_result
from model import calculate_risk


async def main(n: int) -> None:
    await database.init_db()
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n):
        item = FarmRequest(
            farm_id=f"FARM-{i + 100:03d}", region=random.choice(sorted(ALLOWED_REGIONS)),
            crop_type=random.choice(["wheat", "corn", "sunflower", "barley"]),
            area_ha=random.randint(100, 5000), temperature_avg=round(random.uniform(15, 30), 1),
            precipitation_mm=random.randint(40, 500), payment_delay_days=random.randint(0, 90),
            previous_defaults=random.choice([0, 0, 0, 1, 2]), debt=random.randint(0, 9_000_000))
        result = build_result(str(uuid.uuid4()), item, calculate_risk(item), "single")
        ts = now - timedelta(days=random.randint(0, 27), minutes=random.randint(0, 1439))
        result["created_at"] = ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        rows.append(result)
    await database.save_predictions_batch(rows)
    print(f"Добавлено {n} прогнозов в {database.config.DB_PATH}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 120))
