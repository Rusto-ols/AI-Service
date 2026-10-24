from datetime import datetime, timezone

import aiosqlite

import config
from errors import StorageUnavailableError

WEEKDAYS = {"sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3,
            "thursday": 4, "friday": 5, "saturday": 6}

COLUMNS = ("request_id, farm_id, risk_score, risk_level, recommendation, "
           "model_version, source, created_at")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id     TEXT    NOT NULL,
    farm_id        TEXT    NOT NULL,
    risk_score     REAL    NOT NULL,
    risk_level     TEXT    NOT NULL,
    recommendation TEXT    NOT NULL,
    model_version  TEXT    NOT NULL,
    source         TEXT    NOT NULL,
    created_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_request_id ON predictions(request_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _connect():
    return aiosqlite.connect(config.DB_PATH, timeout=5)


def _row(row: aiosqlite.Row) -> dict:
    return dict(row)


async def init_db() -> None:
    try:
        async with _connect() as db:
            await db.execute("PRAGMA journal_mode=WAL")  # читатели и писатель не блокируют друг друга
            await db.executescript(SCHEMA)
            await db.commit()
    except aiosqlite.Error as exc:
        raise StorageUnavailableError() from exc


async def save_prediction(result: dict) -> dict:
    result = {**result, "created_at": result.get("created_at") or utc_now()}
    try:
        async with _connect() as db:
            await db.execute(
                f"INSERT INTO predictions ({COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(result[c.strip()] for c in COLUMNS.split(",")),
            )
            await db.commit()
    except aiosqlite.Error as exc:
        raise StorageUnavailableError() from exc
    return result


async def save_predictions_batch(results: list[dict]) -> list[dict]:
    """Весь пакет — одной транзакцией: executemany + один commit."""
    now = utc_now()
    results = [{**r, "created_at": r.get("created_at") or now} for r in results]
    try:
        async with _connect() as db:
            await db.executemany(
                f"INSERT INTO predictions ({COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [tuple(r[c.strip()] for c in COLUMNS.split(",")) for r in results],
            )
            await db.commit()
    except aiosqlite.Error as exc:
        raise StorageUnavailableError() from exc
    return results


async def get_by_request_id(request_id: str) -> list[dict]:
    try:
        async with _connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT {COLUMNS} FROM predictions WHERE request_id = ? ORDER BY id", (request_id,))
            return [_row(r) for r in await cursor.fetchall()]
    except aiosqlite.Error as exc:
        raise StorageUnavailableError() from exc


async def list_predictions(limit: int, offset: int, risk_level: str | None) -> list[dict]:
    sql = f"SELECT {COLUMNS} FROM predictions"
    params: list = []
    if risk_level is not None:
        sql += " WHERE risk_level = ?"
        params.append(risk_level)
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    try:
        async with _connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(sql, params)
            return [_row(r) for r in await cursor.fetchall()]
    except aiosqlite.Error as exc:
        raise StorageUnavailableError() from exc


async def page_by_weekday(weekday_num: int, page: int, per_page: int) -> tuple[list[dict], int]:
    """Offset-пагинация: COUNT(*) для метаданных + LIMIT/OFFSET для страницы."""
    where = "WHERE CAST(strftime('%w', created_at) AS INTEGER) = ?"
    try:
        async with _connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(f"SELECT COUNT(*) FROM predictions {where}", (weekday_num,))
            total = (await cursor.fetchone())[0]
            cursor = await db.execute(
                f"SELECT {COLUMNS} FROM predictions {where} ORDER BY created_at DESC, id DESC "
                "LIMIT ? OFFSET ?",
                (weekday_num, per_page, (page - 1) * per_page))
            return [_row(r) for r in await cursor.fetchall()], total
    except aiosqlite.Error as exc:
        raise StorageUnavailableError() from exc
