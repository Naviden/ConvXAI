"""FastAPI application for ConvXAI."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from convxai import __version__
from convxai.catalog import DATASET_LABELS
from convxai.dialogue import PROFILES, session_store
from convxai.engine import MODEL_LABELS

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="ConvXAI",
    description="Reproducible conversational explainable AI reference implementation.",
    version=__version__,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class MessageRequest(BaseModel):
    session_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=1_000)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/catalog")
def catalog() -> dict[str, list[str]]:
    return {
        "datasets": list(DATASET_LABELS.values()),
        "models": list(MODEL_LABELS.values()),
        "profiles": list(PROFILES.values()),
    }


@app.post("/api/session")
def create_session() -> dict[str, object]:
    session, response = session_store.create()
    return {"session_id": session.identifier, "response": response.as_dict()}


@app.post("/api/message")
def message(payload: MessageRequest) -> dict[str, object]:
    try:
        response = session_store.respond(payload.session_id, payload.text)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"response": response.as_dict()}
