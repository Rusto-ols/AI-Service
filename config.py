import os


def _env(name: str, default, cast):
    value = os.getenv(name)
    return cast(value) if value is not None else default


DB_PATH = _env("AGRO_DB_PATH", "agro_scoring.db", str)

MAX_BATCH_SIZE = _env("MAX_BATCH_SIZE", 8, int)
MAX_BATCH_WAIT_MS = _env("MAX_BATCH_WAIT_MS", 50, int)
QUEUE_MAX_SIZE = _env("QUEUE_MAX_SIZE", 100, int)
INFERENCE_TIMEOUT_SECONDS = _env("INFERENCE_TIMEOUT_SECONDS", 2.0, float)

MODEL_LATENCY_SECONDS = _env("MODEL_LATENCY_SECONDS", 0.01, float)

LLM_BASE_URL = _env("LLM_BASE_URL", "https://vireonix.ai/v1", str)
LLM_MODEL = _env("LLM_MODEL", "auto", str)
LLM_TIMEOUT_SECONDS = _env("LLM_TIMEOUT_SECONDS", 60.0, float)
