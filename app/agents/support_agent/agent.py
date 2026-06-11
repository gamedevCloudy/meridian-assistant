from datetime import date

from langchain_openrouter import ChatOpenRouter

from app.agents.support_agent.prompt import SYSTEM_PROMPT_TEMPLATE
from app.agents.tools import tools
from app.config import Config

model = ChatOpenRouter(model=Config.DEFAULT_LLM_SM, temperature=0.2, max_retries=3)
model_with_tools = model.bind_tools(tools)


def get_prompt_messages(messages: list, retrieved_context: str = "") -> list:
    """Build the message list sent to the LLM. Prepends system prompt and,
    on the first turn, retrieved RAG context."""
    system = SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())
    if retrieved_context:
        system += f"\n\n[retrieved knowledge — cite these sources]\n{retrieved_context}"
    return [{"role": "system", "content": system}, *messages]
