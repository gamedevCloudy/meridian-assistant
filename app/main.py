import logging
import threading
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage

from app.agents.chat_models import ChatRequest, ChatResponse
from app.agents.graph import agent
from app.data_loader.store import get_vector_store
from app.db import DB_PATH, init_db
from app.history import (
    append_history,
    ensure_history,
    load_history,
)
from app.logger import setup_logging
from app.routers.bookings import router as bookings_router
from app.routers.chat import router as chat_router

setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()


def _warm_vector_store() -> None:
    """Load embeddings + chroma in background so first request is fast."""
    try:
        get_vector_store()
        logger.info("Vector store pre-loaded")
    except Exception:
        logger.exception("Failed to pre-load vector store")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    ensure_history()
    threading.Thread(target=_warm_vector_store, daemon=True).start()
    logger.info("Meridian Assistant started; bookings DB=%s", DB_PATH)
    yield


app = FastAPI(title="Meridian Assistant", lifespan=lifespan)

app.include_router(bookings_router, prefix="/api/v1")
app.include_router(chat_router)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = load_history(req.session_id)
    history.append(HumanMessage(content=req.message))
    append_history(req.session_id, "user", req.message)

    result = agent.invoke({"messages": history})

    final = result["messages"][-1]
    answer = final.content if hasattr(final, "content") else str(final)

    tool_names = []
    for m in result["messages"]:
        if hasattr(m, "tool_calls") and m.tool_calls:
            tool_names.extend(tc["name"] for tc in m.tool_calls)

    handoff = result.get("handoff_requested", False)
    handoff_payload = result.get("handoff_payload") or {}
    handoff_reason = handoff_payload.get("reason") if handoff else None

    append_history(req.session_id, "assistant", answer)

    if handoff:
        logger.info("Handoff emitted: %s | session=%s", handoff_reason, req.session_id)

    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        handoff=handoff,
        handoff_reason=handoff_reason,
        tool_calls=sorted(set(tool_names)),
        llm_calls=result.get("llm_calls", 0),
    )
