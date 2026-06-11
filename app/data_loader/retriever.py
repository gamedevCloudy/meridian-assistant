from app.data_loader.store import get_vector_store


def retrieve(query: str, k: int = 4) -> list:
    return get_vector_store().similarity_search(query, k=k)
