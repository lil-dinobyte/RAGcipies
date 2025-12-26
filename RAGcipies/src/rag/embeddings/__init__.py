from .base import EmbeddingModel, EmbeddingBase
from .fake import FakeEmbeddingModel

# Importación opcional de OpenAI (puede no estar instalado)
try:
    from .openai import OpenAIEmbeddingModel
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAIEmbeddingModel = None  # type: ignore
    _OPENAI_AVAILABLE = False

# Importación opcional de Ollama (puede no estar instalado)
try:
    from .ollama import OllamaEmbeddingModel
    _OLLAMA_AVAILABLE = True
except ImportError:
    OllamaEmbeddingModel = None  # type: ignore
    _OLLAMA_AVAILABLE = False

from .factory import create_embedding_model, EmbeddingBackend

__all__ = [
    "EmbeddingModel",
    "EmbeddingBase",
    "FakeEmbeddingModel",
    "OpenAIEmbeddingModel",
    "OllamaEmbeddingModel",
    "create_embedding_model",
    "EmbeddingBackend",
]