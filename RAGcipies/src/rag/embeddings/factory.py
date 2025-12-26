from enum import Enum
from typing import Type
from .base import EmbeddingModel
from .fake import FakeEmbeddingModel
from .openai import OpenAIEmbeddingModel

# Importación opcional de Ollama embeddings
try:
    from .ollama import OllamaEmbeddingModel
    _OLLAMA_AVAILABLE = True
except ImportError:
    OllamaEmbeddingModel = None  # type: ignore
    _OLLAMA_AVAILABLE = False

class EmbeddingBackend(str, Enum):
    FAKE = "fake"
    OPENAI = "openai"
    OLLAMA = "ollama"  # ✨ NUEVO

# Registry: mapea cada backend a su clase correspondiente (Strategy Pattern)
_EMBEDDING_REGISTRY: dict[EmbeddingBackend, Type[EmbeddingModel]] = {
    EmbeddingBackend.FAKE: FakeEmbeddingModel,
    EmbeddingBackend.OPENAI: OpenAIEmbeddingModel,
}

# Agregar Ollama si está disponible
if _OLLAMA_AVAILABLE:
    _EMBEDDING_REGISTRY[EmbeddingBackend.OLLAMA] = OllamaEmbeddingModel

def create_embedding_model(backend: EmbeddingBackend, **kwargs) -> EmbeddingModel:
    """
    Factory function que crea una instancia del modelo de embeddings
    según el backend especificado usando el patrón Strategy + Registry.
    
    Args:
        backend: El backend a usar (FAKE, OPENAI, OLLAMA, etc.)
        **kwargs: Argumentos adicionales para el constructor 
                  (model, base_url para Ollama, etc.)
        
    Returns:
        Una instancia del modelo de embeddings correspondiente
        
    Raises:
        ValueError: Si el backend no está registrado
    """
    embedding_class = _EMBEDDING_REGISTRY.get(backend)
    
    if embedding_class is None:
        available = ", ".join(b.value for b in _EMBEDDING_REGISTRY.keys())
        raise ValueError(
            f"Invalid embedding backend: {backend.value}. "
            f"Available backends: {available}"
        )
    
    return embedding_class(**kwargs)