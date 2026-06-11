"""
Chat router: streaming + session management.

POST   /chat/stream                 - SSE stream of agent run
GET    /sessions                    - list all sessions
GET    /sessions/{session_id}       - full message history
DELETE /sessions/{session_id}       - delete a session
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.chat_models import ChatRequest
from app.agents.streaming import stream_chat
from app.history import (
    append_history,
    delete_session,
    get_session_messages,
    list_sessions,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n".encode()


async def _event_stream(session_id: str, user_message: str) -> AsyncIterator[bytes]:
    final_answer: str | None = None
    async for ev in stream_chat(session_id, user_message):
        kind = ev["event"]
        if kind == "done":
            final_answer = ev["data"].get("answer") or ""
        yield _sse(kind, ev["data"])

    if final_answer:
        try:
            await asyncio.to_thread(append_history, session_id, "user", user_message)
            await asyncio.to_thread(append_history, session_id, "assistant", final_answer)
        except Exception:
            logger.exception("Failed to persist chat history for session=%s", session_id)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream the agent's response as Server-Sent Events.

    Event types: meta, token, tool_call, tool_result, handoff, done, error.
    `token` events carry a `delta` string the client appends to a markdown
    renderer. The full final text is on the `done` event.
    """
    return StreamingResponse(
        _event_stream(req.session_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/sessions")
def sessions_list():
    return {"sessions": list_sessions()}


@router.get("/sessions/{session_id}")
def session_get(session_id: str):
    msgs = get_session_messages(session_id)
    if not msgs:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "messages": msgs}


@router.delete("/sessions/{session_id}")
def session_delete(session_id: str):
    deleted = delete_session(session_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "deleted": deleted}
