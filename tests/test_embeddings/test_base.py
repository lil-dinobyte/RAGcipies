"""
Tests para la clase base de embeddings.
"""
import pytest

from RAGcipies.src.rag.embeddings.base import EmbeddingModel, EmbeddingBase


def test_embedding_model_es_abstracta():
    with pytest.raises(TypeError):
        EmbeddingModel()
    with pytest.raises(TypeError):
        EmbeddingBase()


def test_fake_embedding_implementa_embedding_model(fake_model):
    assert isinstance(fake_model, EmbeddingModel)
    assert isinstance(fake_model, EmbeddingBase)
    assert hasattr(fake_model, "embed")
    assert callable(getattr(fake_model, "embed"))

