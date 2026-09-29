import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    backend: str = "openai"          # "openai" | "fake" (offline, for tests)
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    chunk_size: int = 700            # chars
    chunk_overlap: int = 100
    top_k: int = 4
    min_score: float = 0.25          # refuse if best cosine is below this
    max_concurrency: int = 5
    rpm: int = 300                   # client-side pacing under provider limit
    db_path: str = "data/capstone.db"
    corpus_dir: str = "data/corpus"


def get_settings(**over) -> Settings:
    load_dotenv()
    s = Settings()
    s.backend = os.getenv("BACKEND", s.backend)
    s.chat_model = os.getenv("CHAT_MODEL", s.chat_model)
    s.embed_model = os.getenv("EMBED_MODEL", s.embed_model)
    s.db_path = os.getenv("DB_PATH", s.db_path)
    s.corpus_dir = os.getenv("CORPUS_DIR", s.corpus_dir)
    if s.backend == "fake":
        s.min_score = 0.12
    for k, v in over.items():
        setattr(s, k, v)
    return s
