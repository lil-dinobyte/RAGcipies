from RAGcipies.src.rag.embeddings.fake import FakeEmbeddingModel


def test_multiple_embeddings_independientes():
    model1 = FakeEmbeddingModel()
    model2 = FakeEmbeddingModel()
    
    text = "test text"
    emb1 = model1.embed(text)
    emb2 = model2.embed(text)
    
    assert emb1 == emb2


def test_embedding_similaridad_basica(fake_model):
    emb1 = fake_model.embed("pollo al curry")
    emb2 = fake_model.embed("pollo con curry")

    similarity = sum(a * b for a, b in zip(emb1, emb2))
    
    assert isinstance(similarity, float)
    assert -1.0 <= similarity <= 1.0

