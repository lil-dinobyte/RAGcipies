import pytest


def test_embed_retorna_lista_de_floats(mock_ollama_post_success, ollama_model):
    result = ollama_model.embed("test text")
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)


def test_embed_dimension_consistente(mock_ollama_post_success, ollama_model):
    texts = ["texto 1", "texto 2", "texto muy largo con muchas palabras"]
    embeddings = [ollama_model.embed(text) for text in texts]
    
    dimensions = [len(emb) for emb in embeddings]
    assert len(set(dimensions)) == 1
    assert dimensions[0] == 768


def test_embed_diferentes_textos_producen_diferentes_embeddings(mock_ollama_post_success, ollama_model, sample_texts):
    embedding1 = ollama_model.embed(sample_texts["pollo"])
    embedding2 = ollama_model.embed(sample_texts["ensalada"])
    
    assert embedding1 != embedding2


def test_embed_texto_con_unicode(mock_ollama_post_success, ollama_model, sample_texts):
    texts = [
        sample_texts["pollo"],
        sample_texts["pollo_emoji"],
        sample_texts["noquis"],
        sample_texts["cafe"]
    ]
    
    for text in texts:
        embedding = ollama_model.embed(text)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


def test_embed_texto_vacio_lanza_error(ollama_model):
    with pytest.raises(ValueError, match="no puede estar vacío"):
        ollama_model.embed("")
    
    with pytest.raises(ValueError, match="no puede estar vacío"):
        ollama_model.embed("   ")


def test_embed_llama_a_ollama_correctamente(mock_ollama_post_success, ollama_model):
    text = "test text"
    ollama_model.embed(text)
    
    mock_ollama_post_success.assert_called_once()
    call_args = mock_ollama_post_success.call_args
    
    assert call_args[0][0] == "http://localhost:11434/api/embeddings"
    assert call_args[1]["json"] == {
        "model": "nomic-embed-text",
        "prompt": text
    }
    assert call_args[1]["timeout"] == 300


def test_embed_maneja_modelo_no_encontrado(mock_ollama_post_error_404, ollama_model):
    with pytest.raises(RuntimeError, match="no encontrado"):
        ollama_model.embed("test text")


def test_embed_maneja_error_conexion(mock_ollama_post_connection_error, ollama_model):
    with pytest.raises(RuntimeError, match="Error al generar embedding con Ollama"):
        ollama_model.embed("test text")


def test_embed_maneja_respuesta_vacia(mock_ollama_post_empty_response, ollama_model):
    with pytest.raises(RuntimeError, match="no contiene un embedding válido"):
        ollama_model.embed("test text")


def test_embed_con_diferentes_modelos(mock_ollama_post_success):
    from RAGcipies.src.rag.embeddings.ollama import OllamaEmbeddingModel
    
    modelos = ["embeddinggemma", "qwen3-embedding", "all-minilm"]
    
    for modelo in modelos:
        model = OllamaEmbeddingModel(model=modelo)
        model.embed("test text")
        
        call_args = mock_ollama_post_success.call_args
        assert call_args[1]["json"]["model"] == modelo


def test_embed_timeout_configurado(mock_ollama_post_success):
    from RAGcipies.src.rag.embeddings.ollama import OllamaEmbeddingModel
    
    model = OllamaEmbeddingModel(timeout=60)
    model.embed("test text")
    
    call_args = mock_ollama_post_success.call_args
    assert call_args[1]["timeout"] == 60
