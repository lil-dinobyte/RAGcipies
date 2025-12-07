from enum import Enum
from typing import Type, Optional
from .base import VectorStore
from .in_memory import InMemoryVectorStore

# Importación opcional de ChromaDB
try:
    from .chromadb_store import ChromaDBVectorStore
    _CHROMADB_AVAILABLE = True
except ImportError:
    ChromaDBVectorStore = None  # type: ignore
    _CHROMADB_AVAILABLE = False


class VectorStoreBackend(str, Enum):
    """Backends disponibles para VectorStore."""
    IN_MEMORY = "in_memory"
    CHROMADB = "chromadb"


# Registry: mapea cada backend a su clase correspondiente (Strategy Pattern)
_VECTOR_STORE_REGISTRY: dict[VectorStoreBackend, Type[VectorStore]] = {
    VectorStoreBackend.IN_MEMORY: InMemoryVectorStore,
}

# Agregar ChromaDB solo si está disponible
if _CHROMADB_AVAILABLE and ChromaDBVectorStore is not None:
    _VECTOR_STORE_REGISTRY[VectorStoreBackend.CHROMADB] = ChromaDBVectorStore


def create_vector_store(
    backend: VectorStoreBackend,
    **kwargs
) -> VectorStore:
    """
    Factory function que crea una instancia del VectorStore
    según el backend especificado usando el patrón Strategy + Registry.
    
    Args:
        backend: El backend a usar (IN_MEMORY, CHROMADB, etc.)
        **kwargs: Parámetros específicos del backend:
            - Para CHROMADB:
                - collection_name: str = "recipes"
                - persist_directory: Optional[str] = None
                - embedding_function: Optional[Callable] = None
            - Para IN_MEMORY: no requiere parámetros adicionales
    
    Returns:
        Una instancia del VectorStore correspondiente
        
    Raises:
        ValueError: Si el backend no está registrado
        
    Examples:
        >>> # InMemory (simple, sin parámetros)
        >>> store = create_vector_store(VectorStoreBackend.IN_MEMORY)
        
        >>> # ChromaDB en memoria
        >>> store = create_vector_store(VectorStoreBackend.CHROMADB)
        
        >>> # ChromaDB con persistencia
        >>> store = create_vector_store(
        ...     VectorStoreBackend.CHROMADB,
        ...     persist_directory="./chroma_db",
        ...     collection_name="recipes"
        ... )
    """
    vector_store_class = _VECTOR_STORE_REGISTRY.get(backend)
    
    if vector_store_class is None:
        available = ", ".join(b.value for b in _VECTOR_STORE_REGISTRY.keys())
        error_msg = (
            f"Invalid vector store backend: {backend.value}. "
            f"Available backends: {available}"
        )
        if backend == VectorStoreBackend.CHROMADB and not _CHROMADB_AVAILABLE:
            error_msg += (
                "\nNote: ChromaDB backend requires 'chromadb' package. "
                "Install it with: pip install chromadb"
            )
        raise ValueError(error_msg)
    
    # Crear instancia con los parámetros específicos del backend
    return vector_store_class(**kwargs)

