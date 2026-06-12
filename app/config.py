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
    AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen/qwen3.6-flash")
    DEFAULT_LLM_MED = os.getenv("LLM_MODEL_MED", AGENT_MODEL)
    DEFAULT_LLM_SM = os.getenv("LLM_MODEL_SM", AGENT_MODEL)
    EMBEDDING_MODEL = "baai/bge-base-en-v1.5"
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    DATA_DIR = str(BASE_DIR / "data")
    CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma_db")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
