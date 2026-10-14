"""Composition root: configuration, lifecycle, routes, and exception handlers only."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.dependencies import build_container
from src.api.errors import register_error_handlers
from src.api.routes import router
from src.core.config import Settings

def create_app(settings: Settings | None = None, container=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        application.state.container = container if container is not None else build_container(settings or Settings())
        yield

    application = FastAPI(title="Grounded Engineering Chatbot", version="1.0.0", lifespan=lifespan)
    register_error_handlers(application)
    application.include_router(router)
    return application

app = create_app()
