import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Real environment variables win over .env.
load_dotenv()

# Model settings. "or": an empty LLM_MODEL= line (copied from .env.example)
# must not win over the default.
MODEL = os.getenv("LLM_MODEL") or "gpt-6-luna"
EMBEDDING_MODEL = "text-embedding-ada-002"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Document chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.7

# Supported file types
SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt"]

NO_KEY_MESSAGE = (
    "🔑 **No API key set.** Add `OPENROUTER_API_KEY` (recommended) or "
    "`OPENAI_API_KEY` to your `.env` file, or paste a key in the "
    "\"API key\" field above."
)


class MissingKeyError(RuntimeError):
    """Raised when no API key is available from the UI, OPENROUTER_API_KEY or OPENAI_API_KEY."""


@dataclass(frozen=True)
class LLMSettings:
    # repr=False: the key must never show up in a log line or traceback.
    api_key: str = field(repr=False)
    base_url: Optional[str]
    model: str
    embedding_model: str


def resolve_llm_settings(session_key: str = "") -> LLMSettings:
    """
    Pick the provider. Order: key pasted in the UI, OPENROUTER_API_KEY (recommended),
    OPENAI_API_KEY, then the old GPT_KEY. Empty values count as unset.
    A pasted key starting with "sk-or-" is an OpenRouter key; any other is OpenAI.
    """
    session_key = (session_key or "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or ""
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_KEY") or ""

    if session_key:
        use_openrouter, key = session_key.startswith("sk-or-"), session_key
    elif openrouter_key:
        use_openrouter, key = True, openrouter_key
    elif openai_key:
        use_openrouter, key = False, openai_key
    else:
        raise MissingKeyError(NO_KEY_MESSAGE)

    if use_openrouter:
        # OpenRouter names models with the provider prefix.
        return LLMSettings(key, OPENROUTER_BASE_URL, _prefixed(MODEL), _prefixed(EMBEDDING_MODEL))
    return LLMSettings(key, None, MODEL, EMBEDDING_MODEL)


def _prefixed(name: str) -> str:
    return name if "/" in name else f"openai/{name}"


def redact(text: str, api_key: str = "") -> str:
    """Hide the key (and anything shaped like one) before text is shown in the UI."""
    if api_key:
        text = text.replace(api_key, "***")
    return re.sub(r"sk-[A-Za-z0-9_\-\.\*]{6,}", "sk-***", text)


def get_chat_model(settings: LLMSettings, temperature: float, streaming: bool = False):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=temperature,
        streaming=streaming,
    )


def get_embeddings(settings: LLMSettings):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        # Send plain text, not token ids: other providers expect text.
        check_embedding_ctx_length=settings.base_url is None,
    )
