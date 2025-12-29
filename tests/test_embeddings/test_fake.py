import pytest
import math



def test_embed_retorna_lista_de_floats(fake_model):
    result = fake_model.embed("test text")
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)


def test_embed_es_determinista(fake_model):
    text = "pollo al curry"
    
    embedding1 = fake_model.embed(text)
    embedding2 = fake_model.embed(text)
    
    assert embedding1 == embedding2


def test_embed_diferentes_textos_producen_diferentes_embeddings(fake_model, sample_texts):
    embedding1 = fake_model.embed(sample_texts["pollo"])
    embedding2 = fake_model.embed(sample_texts["ensalada"])
    
    assert embedding1 != embedding2


def test_embed_esta_normalizado(fake_model):
    embedding = fake_model.embed("test text")
    
    norm = math.sqrt(sum(x * x for x in embedding))
    assert abs(norm - 1.0) < 1e-6


def test_embed_dimension_consistente(fake_model):
    texts = ["texto 1", "texto 2", "texto muy largo con muchas palabras"]
    embeddings = [fake_model.embed(text) for text in texts]
    
    dimensions = [len(emb) for emb in embeddings]
    assert len(set(dimensions)) == 1
    assert dimensions[0] == 8


def test_embed_texto_vacio_funciona(fake_model):
    embedding = fake_model.embed("")
    assert isinstance(embedding, list)
    assert len(embedding) == 8
    embedding2 = fake_model.embed("   ")
    assert isinstance(embedding2, list)
    assert len(embedding2) == 8


def test_embed_texto_con_unicode(fake_model, sample_texts):
    texts = [
        sample_texts["pollo"],
        sample_texts["pollo_emoji"],
        sample_texts["noquis"],
        sample_texts["cafe"]
    ]
    
    for text in texts:
        embedding = fake_model.embed(text)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

