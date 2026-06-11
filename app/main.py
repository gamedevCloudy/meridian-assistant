from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openrouter import ChatOpenRouter

from app.config import Config

load_dotenv()
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/llm-health")
def llm():
    agent = ChatOpenRouter(model=Config.DEFAULT_LLM_SM)

    res = agent.invoke("hi there; return HELLO")

    return {"answer": res.content}
