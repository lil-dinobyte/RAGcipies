from .models import RecipeDocument, Chunk
from .loader import load_recipes_from_json, recipes_to_chunks
from .pipeline import RAGPipeline

# Re-exportar componentes de sub-módulos
from .embeddings import (
    EmbeddingModel,
    EmbeddingBackend,
    create_embedding_model
)
from .vector_store import (
    VectorStore,
    ScoredChunk,
    VectorStoreBackend,
    create_vector_store,
    InMemoryVectorStore,
    ChromaDBVectorStore
)

__all__ = [
    # Models
    "RecipeDocument",
    "Chunk",
    # Loader
    "load_recipes_from_json",
    "recipes_to_chunks",
    # Pipeline
    "RAGPipeline",
    # Embeddings
    "EmbeddingModel",
    "EmbeddingBackend",
    "create_embedding_model",
    # Vector Store
    "VectorStore",
    "ScoredChunk",
    "VectorStoreBackend",
    "create_vector_store",
    "InMemoryVectorStore",
    "ChromaDBVectorStore",
]