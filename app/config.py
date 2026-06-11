"""
1. Configure defualt models
2. Setup paths: chorma_db, files,
3. Other configs: logger conf, etc
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    DEFAULT_LLM_MED = "qwen/qwen3.6-flash"
    DEFAULT_LLM_SM = "qwen/qwen3.6-flash"
    EMBEDDING_MODEL = "baai/bge-base-en-v1.5"
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    DATA_DIR = str(BASE_DIR / "data")
    CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma_db")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
