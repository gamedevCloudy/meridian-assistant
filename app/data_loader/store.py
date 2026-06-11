import threading

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Config

_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: Chroma | None = None
_chroma_lock = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    return _embeddings


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        with _chroma_lock:
            # Double-check after acquiring lock
            if _vector_store is None:
                _vector_store = Chroma(
                    collection_name="meridian_knowledge_base",
                    embedding_function=get_embeddings(),
                    persist_directory=Config.CHROMA_DB_PATH,
                )
    return _vector_store
