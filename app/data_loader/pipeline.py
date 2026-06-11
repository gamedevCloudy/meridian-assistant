import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Config
from app.logger import setup_logging
from app.data_loader.store import get_vector_store

logger = logging.getLogger(__name__)

DATA_DIR = Path(Config.DATA_DIR)
EXCLUDE_DIRS = {"_eval_data"}

DOC_TYPE_MAP: dict[str, str] = {
    "faqs": "faq",
    "pricing": "pricing",
    "service-areas": "service_area",
    "tnc": "terms",
}


def _infer_doc_type(file_path: Path) -> str:
    return DOC_TYPE_MAP.get(file_path.parent.name, "general")


def _infer_doc_name(file_path: Path) -> str:
    name = file_path.stem
    if "_" in name and name.split("_", 1)[0].isdigit():
        return name.split("_", 1)[1].replace("_", " ").title()
    return name.replace("_", " ").title()


def load_pdf(file_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    doc_type = _infer_doc_type(file_path)
    doc_name = _infer_doc_name(file_path)
    for doc in docs:
        doc.metadata.update({
            "doc_type": doc_type,
            "doc_name": doc_name,
            "source": str(file_path.relative_to(DATA_DIR)),
        })
    return docs


def load_all_pdfs() -> list[Document]:
    docs: list[Document] = []
    for pdf_path in DATA_DIR.rglob("*.pdf"):
        if any(part in EXCLUDE_DIRS for part in pdf_path.relative_to(DATA_DIR).parts):
            continue
        docs.extend(load_pdf(pdf_path))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    return splitter.split_documents(docs)


def _doc_id(doc: Document) -> str:
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def store_documents(docs: list[Document]) -> int:
    store = get_vector_store()
    existing_ids = set(store.get(include=[])["ids"])
    new_docs = [d for d in docs if _doc_id(d) not in existing_ids]
    if new_docs:
        store.add_documents(new_docs, ids=[_doc_id(d) for d in new_docs])
        logger.info("Added %d new chunks", len(new_docs))
    else:
        logger.info("No new chunks to add")
    return len(new_docs)


def run_pipeline() -> int:
    setup_logging()
    raw_docs = load_all_pdfs()
    logger.info("Loaded %d raw document pages", len(raw_docs))
    split_docs = split_documents(raw_docs)
    logger.info("Split into %d chunks", len(split_docs))
    added = store_documents(split_docs)
    logger.info("Total chunks in DB: %d", len(get_vector_store().get(include=[])["ids"]))
    return added
