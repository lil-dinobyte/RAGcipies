import pytest
from unittest.mock import Mock, patch
import hashlib
from RAGcipies.src.rag.embeddings.fake import FakeEmbeddingModel
from RAGcipies.src.rag.embeddings.ollama import OllamaEmbeddingModel


@pytest.fixture
def fake_model():
    return FakeEmbeddingModel()


@pytest.fixture
def sample_texts():
    return {
        "pollo": "pollo al curry",
        "ensalada": "ensalada de frutas",
        "arroz": "arroz con verduras salteadas",
        "pollo_emoji": "pollo al curry 🍗",
        "noquis": "receta con ñoquis",
        "cafe": "café con leche"
    }



@pytest.fixture
def mock_ollama_post_success():
    with patch('RAGcipies.src.rag.embeddings.ollama.requests.post') as mock_post:
        def side_effect(*args, **kwargs):
            text = kwargs.get('json', {}).get('prompt', '')
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            embedding = [(hash_int % 1000 + i) / 10000.0 for i in range(768)]
            
            mock_response = Mock()
            mock_response.json.return_value = {"embedding": embedding}
            mock_response.raise_for_status = Mock()
            return mock_response
        
        mock_post.side_effect = side_effect
        yield mock_post


@pytest.fixture
def mock_ollama_post_error_404():
    with patch('RAGcipies.src.rag.embeddings.ollama.requests.post') as mock_post:
        from requests.exceptions import HTTPError
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_post.return_value = mock_response
        mock_post.side_effect = HTTPError("404 Not Found")
        yield mock_post


@pytest.fixture
def mock_ollama_post_connection_error():
    with patch('RAGcipies.src.rag.embeddings.ollama.requests.post') as mock_post:
        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError("Connection refused")
        yield mock_post


@pytest.fixture
def mock_ollama_post_empty_response():
    with patch('RAGcipies.src.rag.embeddings.ollama.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def ollama_model():
    return OllamaEmbeddingModel(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

