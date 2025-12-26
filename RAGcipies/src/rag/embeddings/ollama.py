from typing import List, Optional
import os
import requests
from .base import EmbeddingModel


class OllamaEmbeddingModel(EmbeddingModel):
    """
    Embedding usando Ollama (modelos locales).
    No requiere API key, funciona completamente offline.
    
    Referencias:
    - https://docs.ollama.com/capabilities/embeddings
    - Modelos recomendados: embeddinggemma, qwen3-embedding, all-minilm, nomic-embed-text
    """
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 300
    ):
        """
        Args:
            model: Modelo de embeddings de Ollama a usar.
                   Opciones recomendadas:
                   - "nomic-embed-text" (768 dims, recomendado)
                   - "embeddinggemma" 
                   - "qwen3-embedding"
                   - "all-minilm"
            base_url: URL base de la API de Ollama
            timeout: Timeout en segundos para las peticiones HTTP (default: 300 = 5 minutos)
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # No verificamos aquí, dejamos que falle en embed() con mejor mensaje
    
    def embed(self, text: str) -> List[float]:
        """
        Genera un embedding para el texto dado usando Ollama.
        
        Args:
            text: Texto a convertir en embedding
            
        Returns:
            Lista de floats representando el vector de embedding (L2-normalizado)
            
        Raises:
            ValueError: Si el texto está vacío
            RuntimeError: Si hay un error al llamar a Ollama o el modelo no está descargado
        """
        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Ollama retorna un diccionario con la clave "embedding"
            embedding = result.get("embedding", [])
            if not embedding:
                raise RuntimeError("La respuesta de Ollama no contiene un embedding válido")
            
            return embedding
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            
            # Mensaje más claro si el modelo no está descargado
            if "not found" in error_msg.lower() or "404" in error_msg.lower():
                raise RuntimeError(
                    f"Modelo '{self.model}' no encontrado en Ollama.\n"
                    f"Descárgalo primero con: ollama pull {self.model}\n"
                    f"Modelos recomendados: nomic-embed-text, embeddinggemma, qwen3-embedding, all-minilm"
                )
            else:
                raise RuntimeError(
                    f"Error al generar embedding con Ollama: {error_msg}\n"
                    f"Asegúrate de que:\n"
                    f"  1. Ollama esté corriendo (ollama serve)\n"
                    f"  2. El modelo {self.model} esté descargado (ollama pull {self.model})"
                )
        except Exception as e:
            raise RuntimeError(
                f"Error inesperado al generar embedding con Ollama: {e}\n"
                f"Asegúrate de que Ollama esté corriendo en {self.base_url}"
            )