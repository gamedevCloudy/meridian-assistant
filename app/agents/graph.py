"""
LangGraph agent for the Meridian Assistant.

Flow:
  START
    -> retrieve   (first turn only, populates retrieved_context)
    -> llm_call   (decides: tool call, final answer, or handoff)
    -> [conditional]
         tool_calls present  -> tool_node -> llm_call (loop, max 5)
         no tool_calls       -> END

State carries the full message history plus retrieved_context and a
handoff_requested flag so the /chat endpoint can decide how to respond.
"""

import json
import logging
import operator
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agents.support_agent.agent import get_prompt_messages, model_with_tools
from app.agents.tools import tools_by_name
from app.data_loader.retriever import retrieve

logger = logging.getLogger(__name__)

MAX_LLM_CALLS = 5


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    retrieved_context: str
    llm_calls: int
    handoff_requested: bool
    handoff_payload: dict[str, Any]


def _text_content(content: Any) -> str:
    """langchain's HumanMessage.content is typed as str | list[str | dict] for
    multimodal support, but our chat endpoint always sends plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and isinstance(content[0], str):
        return " ".join(content)
    return str(content)


def retrieve_node(state: AgentState) -> dict:
    """First-turn retrieval. Skips if context already populated (follow-up
    turns rely on the LLM calling retrieve_kb tool when needed)."""
    if state.get("retrieved_context"):
        return {}
    last_user = next(
        (m for m in reversed(state["messages"]) if m.type == "human"), None
    )
    if last_user is None:
        return {}
    docs = retrieve(_text_content(last_user.content), k=4)
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('doc_name', 'unknown')} | p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )
    logger.info("Retrieved %d chunks for first-turn query", len(docs))
    return {"retrieved_context": context}


def llm_call_node(state: AgentState) -> dict:
    """Single LLM invocation with the system prompt and message history."""
    messages = get_prompt_messages(state["messages"], state.get("retrieved_context", ""))
    response = model_with_tools.invoke(messages)
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: AgentState) -> dict:
    """Execute every tool call the LLM made on its last message. If
    handoff_to_human is among them, mark the state and short-circuit."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {}
    out: list[ToolMessage] = []
    handoff_payload: dict[str, Any] | None = state.get("handoff_payload")
    handoff_flag: bool = state.get("handoff_requested", False)

    for tc in last.tool_calls:
        tool = tools_by_name.get(tc["name"])
        if tool is None:
            out.append(ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]))
            continue
        try:
            observation = tool.invoke(tc["args"])
        except Exception as e:
            logger.exception("Tool %s failed", tc["name"])
            observation = f"Tool error: {e}"
        out.append(ToolMessage(content=observation, tool_call_id=tc["id"]))

        if tc["name"] == "handoff_to_human":
            handoff_flag = True
            try:
                handoff_payload = json.loads(observation)
            except Exception:
                handoff_payload = {"reason": tc["args"].get("reason"), "context": tc["args"].get("context")}

    return {
        "messages": out,
        "handoff_requested": handoff_flag,
        "handoff_payload": handoff_payload,
    }


def should_continue(state: AgentState) -> str:
    """Decide whether to loop back through the tool node or terminate.
    Returns the next node name or langgraph's END sentinel.
    """
    if state.get("handoff_requested"):
        return END
    if state.get("llm_calls", 0) >= MAX_LLM_CALLS:
        logger.warning("LLM call cap hit; terminating")
        return END
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_node"
    return END


def build_graph():
    # The AgentState TypedDict satisfies the runtime contract; the langgraph
    # `StateT` bound uses a strict Protocol that ty cannot prove structurally
    # for TypedDict subclasses with reducer-annotated fields.
    builder = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("llm_call", llm_call_node)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")

    return builder.compile()


agent = build_graph()
