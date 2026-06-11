import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openrouter import ChatOpenRouter

from app.config import Config
from app.data_loader.retriever import retrieve
from app.data_loader.store import get_vector_store
from app.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/llm-health")
def llm():
    query = "what are the plumbing service charges?"

    logger.info("Loading vector DB")
    store = get_vector_store()

    logger.info("Retrieving context for query: %s", query)
    docs = retrieve(query, k=3)
    logger.info("Retrieved %d relevant chunks", len(docs))

    context = "\n\n".join(
        f'[{d.metadata["doc_name"]}] {d.page_content}' for d in docs
    )

    prompt = f"""You are a Meridian Home Services assistant. Answer using the context below.

Context:
{context}

Question: {query}

Answer:"""

    agent = ChatOpenRouter(model=Config.DEFAULT_LLM_SM)
    res = agent.invoke(prompt)

    return {"answer": res.content, "sources": [d.metadata["doc_name"] for d in docs]}
