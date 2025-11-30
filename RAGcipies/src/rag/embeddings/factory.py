from enum import Enum
from typing import Type
from .base import EmbeddingModel
from .fake import FakeEmbeddingModel
from .openai import OpenAIEmbeddingModel

class EmbeddingBackend(str, Enum):
    FAKE = "fake"
    OPENAI = "openai"

# Registry: mapea cada backend a su clase correspondiente (Strategy Pattern)
_EMBEDDING_REGISTRY: dict[EmbeddingBackend, Type[EmbeddingModel]] = {
    EmbeddingBackend.FAKE: FakeEmbeddingModel,
    EmbeddingBackend.OPENAI: OpenAIEmbeddingModel,
}

def create_embedding_model(backend: EmbeddingBackend) -> EmbeddingModel:
    """
    Factory function que crea una instancia del modelo de embeddings
    según el backend especificado usando el patrón Strategy + Registry.
    
    Args:
        backend: El backend a usar (FAKE, OPENAI, etc.)
        
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
    
    return embedding_class()

