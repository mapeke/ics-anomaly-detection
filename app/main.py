"""FastAPI entry point for the external-validation web app."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import router

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="ICS Anomaly Detection — External Validation",
        version="0.1.0",
        description=(
            "Upload an unseen Morris gas-pipeline ARFF/CSV and score it against "
            "a trained artifact. Localhost-only, no auth."
        ),
    )
    app.include_router(router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def root() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
