"""FastAPI Application entry point with CORS, lifespan handlers, and routing."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.config import get_settings
from app.db.database import init_db
from app.services.embeddings.embedder import get_embedder
from app.services.ml.predictor import get_cached_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_resume_system")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()

    logger.info("Pre-warming embedding model...")
    embedder = get_embedder()
    logger.info("Embedding backend active: %s (dim: %d)", embedder.__class__.__name__, embedder.dimension)

    logger.info("Loading ML classification models...")
    metrics = get_cached_metrics()
    logger.info("Best ML model: %s", metrics.selected_model)

    yield

    logger.info("Shutting down AI Resume Screening API...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI-Based Resume Screening and Candidate Evaluation System",
        description=(
            "Production-quality academic candidate evaluation system combining traditional ATS, "
            "sentence embeddings (BAAI/bge-small-en-v1.5), FAISS vector search, machine learning "
            "(LR/RF/KNN), Gemini GenAI, and LangGraph multi-agent orchestration."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = [
        settings.frontend_origin,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
