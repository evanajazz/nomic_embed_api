"""Nomic Embedding microservice

Exposes two endpoints:
  POST /embed  – returns embeddings for a batch of texts.
  GET  /health – simple liveness probe.

Environment variables:
  * NOMIC_API_KEY    – required, your Nomic account key.
  * MODEL_NAME       – optional, default "nomic-embed-text-v1".
  * DEFAULT_TASK     – optional, default "search_document".
  * DIMENSIONALITY   – optional, integer, only applied if model supports it (v1.5+).

Start with
    uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
"""

from __future__ import annotations

import logging
import os
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

import nomic
from nomic import embed

# ---------------------------------------------------------------------------
# Config --------------------------------------------------------------------
# ---------------------------------------------------------------------------
API_KEY = os.getenv("NOMIC_API_KEY")
if not API_KEY:
    raise RuntimeError("NOMIC_API_KEY not set in environment")

nomic.login(API_KEY)

MODEL_NAME = os.getenv("MODEL_NAME", "nomic-embed-text-v1")
DEFAULT_TASK: Literal[
    "search_document", "search_query", "classification", "clustering"
] = os.getenv("DEFAULT_TASK", "search_document")  # type: ignore[assignment]

DIMENSIONALITY_ENV = os.getenv("DIMENSIONALITY")
DIMENSIONALITY = int(DIMENSIONALITY_ENV) if DIMENSIONALITY_ENV else None

ALLOWED_TASKS = {
    "search_document",
    "search_query",
    "classification",
    "clustering",
}

# ---------------------------------------------------------------------------
# Logging -------------------------------------------------------------------
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info("Using model %s (default task %s)", MODEL_NAME, DEFAULT_TASK)

# ---------------------------------------------------------------------------
# FastAPI app ----------------------------------------------------------------
# ---------------------------------------------------------------------------
app = FastAPI(title="Nomic Embedding Service", version="1.0.0")


class EmbedRequest(BaseModel):
    """Request body for /embed"""

    text: List[str] = Field(..., description="List of strings to embed")
    task_type: Optional[Literal[
        "search_document", "search_query", "classification", "clustering"
    ]] = Field(None, description="Embedding task type")
    dimensionality: Optional[int] = Field(
        None, ge=64, le=768, description="Desired embedding dimensionality (v1.5+)"
    )

    # Pydantic validation -----------------------------------------------------
    @validator("text")
    def _strip_and_check(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("'text' must contain at least one string")
        return [t.strip() for t in v]

    @validator("task_type")
    def _validate_task(cls, v):
        if v is not None and v not in ALLOWED_TASKS:
            raise ValueError(f"task_type must be one of {sorted(ALLOWED_TASKS)}")
        return v


@app.post("/embed")
async def get_embed(req: EmbedRequest):
    task = req.task_type or DEFAULT_TASK
    dim = req.dimensionality or DIMENSIONALITY

    if task not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task_type: {task}")

    # Nomic v1 ignores dimensionality; warn the user.
    if dim is not None and "v1.5" not in MODEL_NAME:
        logger.warning("Dimensionality ignored by model %s", MODEL_NAME)
        dim = None

    logger.info("Embedding %d texts with task=%s, model=%s, dim=%s", len(req.text), task, MODEL_NAME, dim)

    vectors = embed.text(
        texts=req.text,
        model=MODEL_NAME,
        task_type=task,
        dimensionality=dim,  # None is safe
    )["embeddings"]

    return {"vectors": vectors}


@app.get("/health")
async def health_check():
    return {"status": "ok"}












