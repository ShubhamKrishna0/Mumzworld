"""FastAPI application exposing the Moms Verdict pipeline as a REST endpoint."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.errors import (
    EmptyInputError,
    GenerationError,
    InsufficientDataError,
    ParseError,
    ValidationError,
)
from app.pipeline import run_pipeline
from app.schema import MomsVerdict

app = FastAPI(title="Moms Verdict Generator")


@app.post("/verdict", response_model=MomsVerdict)
async def generate_verdict(file: UploadFile = File(...)):
    """Accept a JSON or TXT file of reviews and return a MomsVerdict JSON response."""
    raw_data = await file.read()

    # Determine content type from the upload metadata or filename extension
    content_type = file.content_type or ""
    if not content_type or content_type == "application/octet-stream":
        filename = file.filename or ""
        if filename.endswith(".json"):
            content_type = "application/json"
        elif filename.endswith(".txt"):
            content_type = "text/plain"

    try:
        verdict = run_pipeline(raw_data, content_type)
    except (EmptyInputError, ParseError, InsufficientDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (GenerationError, ValidationError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

    return verdict
