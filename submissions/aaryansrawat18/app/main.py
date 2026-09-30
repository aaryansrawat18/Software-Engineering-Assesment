from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db import init_db
from app.logging import configure_logging
from app.routers.auth import router as auth_router
from app.routers.documents import router as documents_router
from app.services.storage import ensure_upload_directory


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the database tables and the shared upload folder when the API starts."""
    init_db()
    ensure_upload_directory()
    yield


def create_app() -> FastAPI:
    get_settings()
    configure_logging()
    app = FastAPI(title="DocuMind", lifespan=lifespan)
    app.include_router(auth_router)
    app.include_router(documents_router)
    return app


app = create_app()
