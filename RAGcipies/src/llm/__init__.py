from .base import LLMClient, LLMBase
from .dummy import DummyLLM

# Importación opcional de OpenAI (puede no estar instalado)
try:
    from .openai import OpenAILLM
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAILLM = None  # type: ignore
    _OPENAI_AVAILABLE = False

# Importación opcional de Ollama (puede no estar instalado)
try:
    from .ollama import OllamaLLM
    _OLLAMA_AVAILABLE = True
except ImportError:
    OllamaLLM = None  # type: ignore
    _OLLAMA_AVAILABLE = False

from .factory import create_llm_client, LLMBackend

__all__ = [
    "LLMClient",
    "LLMBase",
    "DummyLLM",
    "OpenAILLM",
    "OllamaLLM",
    "create_llm_client",
    "LLMBackend",
]