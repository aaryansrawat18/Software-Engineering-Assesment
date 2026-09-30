from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db import init_db
from app.logging import configure_logging
from app.routers.auth import router as auth_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


def create_app() -> FastAPI:
    get_settings()
    configure_logging()
    app = FastAPI(title="DocuMind", lifespan=lifespan)
    app.include_router(auth_router)
    return app


app = create_app()
