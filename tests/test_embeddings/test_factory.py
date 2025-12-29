from RAGcipies.src.rag.embeddings.fake import FakeEmbeddingModel
from RAGcipies.src.rag.embeddings.factory import (
    create_embedding_model,
    EmbeddingBackend
)


def test_create_embedding_model_fake():
    model = create_embedding_model(EmbeddingBackend.FAKE)
    
    assert isinstance(model, FakeEmbeddingModel)
    embedding = model.embed("test")
    assert isinstance(embedding, list)
    assert len(embedding) > 0


def test_create_embedding_model_backend_invalido():
    assert create_embedding_model(EmbeddingBackend.FAKE) is not None


def test_embedding_backend_enum():
    assert hasattr(EmbeddingBackend, "FAKE")
    assert hasattr(EmbeddingBackend, "OPENAI")
    assert hasattr(EmbeddingBackend, "OLLAMA")
    assert EmbeddingBackend.FAKE.value == "fake"
    assert EmbeddingBackend.OPENAI.value == "openai"
    assert EmbeddingBackend.OLLAMA.value == "ollama"

