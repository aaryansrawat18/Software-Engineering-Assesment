from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import get_settings
from app.db import init_db
from app.logging import add_request_logging, configure_logging
from app.routers.auth import router as auth_router
from app.routers.documents import router as documents_router
from app.routers.health import router as health_router
from app.routers.questions import router as questions_router
from app.services.storage import ensure_upload_directory

# The one-page UI lives next to the app package: submissions/.../ui/index.html
UI_INDEX_PATH = Path(__file__).resolve().parents[1] / "ui" / "index.html"


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
    add_request_logging(app)
    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(questions_router)
    app.include_router(health_router)

    @app.get("/", include_in_schema=False)
    def home_page() -> FileResponse:
        """Serve the signup, upload, and question page. Swagger stays at /docs."""
        return FileResponse(UI_INDEX_PATH)

    return app


app = create_app()
