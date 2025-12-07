from .base import VectorStore, ScoredChunk
from .in_memory import InMemoryVectorStore

# Importación opcional de ChromaDB (puede no estar instalado)
try:
    from .chromadb_store import ChromaDBVectorStore
    _CHROMADB_AVAILABLE = True
except ImportError:
    ChromaDBVectorStore = None  # type: ignore
    _CHROMADB_AVAILABLE = False

from .factory import create_vector_store, VectorStoreBackend

__all__ = [
    "VectorStore",
    "ScoredChunk",
    "InMemoryVectorStore",
    "ChromaDBVectorStore",
    "create_vector_store",
    "VectorStoreBackend",
]

