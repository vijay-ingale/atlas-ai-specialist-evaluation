import logging
from uuid import uuid4
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from src.core.errors import ApplicationError

logger = logging.getLogger("support_classifier")

def register_error_handlers(app):
    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def error_response(request, status, code, message):
        identifier = getattr(request.state, "request_id", uuid4().hex)
        return JSONResponse(status_code=status, content={"code": code, "message": message,
            "request_id": identifier}, headers={"X-Request-ID": identifier})

    @app.exception_handler(ApplicationError)
    async def application_error(request: Request, exc: ApplicationError):
        # No prompts, document content, keys, or raw provider exceptions in logs.
        logger.warning("request_id=%s error_code=%s", request.state.request_id, exc.code)
        return error_response(request, exc.status_code, exc.code, str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error_response(request, 422, "request_validation_error", "Request does not match the required schema.")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.error("request_id=%s exception_type=%s", getattr(request.state, "request_id", "unknown"), type(exc).__name__)
        return error_response(request, 500, "internal_error", "An unexpected error occurred.")
