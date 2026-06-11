"""
Streaming adapter for the LangGraph agent.

Yields a stream of structured events the router serialises as SSE.
Events:
    meta         - one, at start (session_id)
    tool_call    - LLM invoked a tool
    tool_result  - tool returned
    token        - incremental assistant text (markdown)
    handoff      - handoff_to_human fired
    done         - final summary (last assistant text, tool_calls, llm_calls, handoff)
    error        - exception in the graph
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.graph import agent
from app.history import load_history

logger = logging.getLogger(__name__)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


async def stream_chat(session_id: str, user_message: str) -> AsyncIterator[dict]:
    """Run the graph and yield structured events.

    The first human message in the stream is the new user turn; prior
    context is loaded from history.
    """
    history = load_history(session_id)
    history.append(HumanMessage(content=user_message))
    input_state = {"messages": history}

    yield {"event": "meta", "data": {"session_id": session_id}}

    full_text_parts: list[str] = []
    tool_call_names: list[str] = []
    handoff_flag = False
    handoff_payload: dict[str, Any] | None = None
    llm_calls = 0
    final_text = ""

    try:
        async for ev in agent.astream_events(input_state, version="v2"):
            kind = ev["event"]
            name = ev.get("name", "")

            if kind == "on_chat_model_start":
                llm_calls += 1

            elif kind == "on_chat_model_stream":
                chunk = ev["data"].get("chunk")
                if chunk is None:
                    continue
                # AIMessageChunk.content may be str or list; we send plain text
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    full_text_parts.append(content)
                    yield {"event": "token", "data": {"delta": content}}

            elif kind == "on_chat_model_end":
                output = ev["data"].get("output")
                if isinstance(output, AIMessage):
                    if output.tool_calls:
                        for tc in output.tool_calls:
                            tool_call_names.append(tc["name"])
                            yield {
                                "event": "tool_call",
                                "data": {
                                    "name": tc["name"],
                                    "args": tc.get("args", {}),
                                    "id": tc.get("id"),
                                },
                            }
                    if output.content:
                        text = (
                            output.content
                            if isinstance(output.content, str)
                            else "".join(
                                p if isinstance(p, str) else str(p)
                                for p in output.content
                            )
                        )
                        if text and not full_text_parts:
                            full_text_parts.append(text)
                            yield {"event": "token", "data": {"delta": text}}
                        final_text = text

            elif kind == "on_tool_end":
                output = ev["data"].get("output")
                tool_name = ev.get("name", name)
                if isinstance(output, ToolMessage):
                    tool_name = output.name or tool_name
                    raw = output.content
                else:
                    raw = output
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        payload = parsed
                    except Exception:
                        payload = raw
                else:
                    payload = raw
                if tool_name == "handoff_to_human":
                    handoff_flag = True
                    if isinstance(payload, dict):
                        handoff_payload = {
                            "reason": payload.get("reason"),
                            "context": payload.get("context"),
                        }
                    transfer_msg = "I understand this is a sensitive situation. I'm transferring you to a human agent right away."
                    full_text_parts.append(transfer_msg)
                    yield {"event": "token", "data": {"delta": transfer_msg}}
                    yield {
                        "event": "handoff",
                        "data": handoff_payload or {"reason": None, "context": None},
                    }
                else:
                    yield {
                        "event": "tool_result",
                        "data": {"name": tool_name, "result": payload},
                    }

        if not final_text and full_text_parts:
            final_text = "".join(full_text_parts)

        yield {
            "event": "done",
            "data": {
                "answer": final_text,
                "handoff": handoff_flag,
                "handoff_reason": (handoff_payload or {}).get("reason"),
                "tool_calls": sorted(set(tool_call_names)),
                "llm_calls": llm_calls,
            },
        }
    except Exception as e:
        logger.exception("Stream failed for session=%s", session_id)
        yield {"event": "error", "data": {"message": str(e)}}
