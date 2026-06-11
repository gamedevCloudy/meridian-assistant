import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = ChatOpenRouter(model="auto")

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/llm-health")
def llm():
    agent = ChatOpenRouter(model="google/gemma-4-31b-it:free")

    res = agent.invoke("hi there; return HELLO")

    return {"answer": res.content}
