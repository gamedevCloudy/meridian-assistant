"""
Pydantic models for the /chat endpoint.
"""

from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str
    customer_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    handoff: bool = False
    handoff_reason: Optional[str] = None
    tool_calls: list[str] = []
    llm_calls: int = 0
