import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

FREE_MODEL_PREFERENCES = [
    "z-ai/glm",
    "nvidia/nemotron-3-super",
    "google/gemma-4-31b",
    "thinkingmachines/inkling:",
    "nvidia/nemotron-3-ultra",
    "nvidia/nemotron-3-nano-30b",
    "google/gemma-4-26b",
    "poolside/laguna-s",
    "dots-studio/dots-3",
    "thinkingmachines/inkling-small",
    "nvidia/nemotron-nano-9b-v2",
]

DEPRIORITY_SUBSTRINGS = ["-code", "-vl", "omni", "safety"]


@dataclass
class Settings:
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "")
    site_url: str = os.getenv("SITE_URL", "http://localhost:8000")
    site_name: str = os.getenv("SITE_NAME", "AI Legal Citation Researcher")
    db_path: str = os.getenv("DB_PATH", "data/legal_research.db")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "3000"))
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "120"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    model_preferences: list = field(default_factory=lambda: list(FREE_MODEL_PREFERENCES))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openrouter_api_key)


settings = Settings()
