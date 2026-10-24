import time

import httpx

import config
from errors import LLMProviderError, LLMTimeoutError, LLMUnavailableError


class LLMClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def chat(self, messages: list[dict], model: str | None = None,
                   temperature: float = 0.3, max_tokens: int = 300) -> dict:
        payload = {"model": model or config.LLM_MODEL, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        start = time.perf_counter()
        try:
            response = await self.client.post("/chat/completions", json=payload,
                                              timeout=config.LLM_TIMEOUT_SECONDS)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError() from exc
        except httpx.RequestError as exc:
            raise LLMUnavailableError(f"LLM provider is unavailable: {type(exc).__name__}") from exc
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        if response.status_code == 429 or response.status_code >= 500:
            raise LLMUnavailableError(f"LLM provider is unavailable: HTTP {response.status_code}")
        if response.status_code >= 400:
            raise LLMProviderError(f"LLM provider error: HTTP {response.status_code}")
        try:
            data = response.json()
            choice = data["choices"][0]
            answer = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError() from exc
        if not isinstance(answer, str) or not answer.strip():
            raise LLMProviderError("LLM provider returned an empty answer")
        return {"answer": answer.strip(), "returned_model": data.get("model"),
                "finish_reason": choice.get("finish_reason"), "latency_ms": latency_ms}
