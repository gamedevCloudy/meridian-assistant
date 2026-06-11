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
    DEFAULT_LLM_MED = "nvidia/nemotron-3-super-120b-a12b:free"
    DEFAULT_LLM_SM = "openrouter/free"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DATA_DIR = str(BASE_DIR / "data")
    CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma_db")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
