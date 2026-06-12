import os
import threading

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from app.config import Config


class OpenRouterEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI(
            base_url=Config.OPENROUTER_BASE_URL,
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_embeddings: OpenRouterEmbeddings | None = None
_vector_store: Chroma | None = None
_chroma_lock = threading.Lock()


def get_embeddings() -> OpenRouterEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenRouterEmbeddings()
    return _embeddings


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        with _chroma_lock:
            if _vector_store is None:
                _vector_store = Chroma(
                    collection_name="meridian_knowledge_base",
                    embedding_function=get_embeddings(),
                    persist_directory=Config.CHROMA_DB_PATH,
                )
    return _vector_store


def get_document_count() -> int:
    store = get_vector_store()
    try:
        return len(store.get(include=[])["ids"])
    except Exception:
        return 0


def reset_vector_store() -> None:
    global _vector_store
    with _chroma_lock:
        _vector_store = None
