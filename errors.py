import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("agro_api")


class AppError(Exception):
    status_code = 500
    detail = "Internal server error"

    def __init__(self, detail: str | None = None):
        if detail is not None:
            self.detail = detail
        super().__init__(self.detail)


class BusinessRuleError(AppError):
    status_code = 400
    detail = "Business rule violated"


class NotFoundError(AppError):
    status_code = 404
    detail = "Prediction not found"


class AmbiguousResultError(AppError):
    status_code = 409
    detail = "request_id belongs to a batch; use GET /requests/{request_id}/predictions"


class QueueFullError(AppError):
    status_code = 429
    detail = "Inference queue is full, retry later"


class LLMProviderError(AppError):
    status_code = 502
    detail = "LLM provider returned an invalid response"


class ServiceUnavailableError(AppError):
    status_code = 503
    detail = "Service dependency is unavailable"


class StorageUnavailableError(ServiceUnavailableError):
    detail = "Prediction storage is unavailable"


class LLMUnavailableError(ServiceUnavailableError):
    detail = "LLM provider is unavailable"


class InferenceTimeoutError(AppError):
    status_code = 504
    detail = "Model inference timed out"


class LLMTimeoutError(AppError):
    status_code = 504
    detail = "LLM request timed out"


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _json(request: Request, status_code: int, detail, headers: dict | None = None) -> JSONResponse:
    rid = _request_id(request)
    headers = dict(headers or {})
    if rid:
        headers["X-Request-ID"] = rid
    return JSONResponse(status_code=status_code, content={"detail": detail, "request_id": rid},
                        headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.warning("request_id=%s error=%s status=%s detail=%s", _request_id(request),
                       type(exc).__name__, exc.status_code, exc.detail)
        headers = {"Retry-After": "1"} if exc.status_code == 429 else None
        return _json(request, exc.status_code, exc.detail, headers)

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        return _json(request, exc.status_code, exc.detail, getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = [{k: v for k, v in err.items() if k != "url"} for err in exc.errors()]
        for err in errors:
            if "ctx" in err:
                err["ctx"] = {k: str(v) for k, v in err["ctx"].items()}
        return _json(request, 422, errors)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("request_id=%s unhandled error", _request_id(request))
        return _json(request, 500, "Internal server error")
