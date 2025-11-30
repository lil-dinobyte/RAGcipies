from typing import List
import os
from openai import OpenAI
from .base import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    """
    Embedding usando la API de OpenAI.
    Requiere OPENAI_API_KEY en las variables de entorno.
    
    Referencias:
    - https://platform.openai.com/docs/api-reference/embeddings
    - https://platform.openai.com/docs/guides/embeddings
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Args:
            model: Modelo de embeddings a usar.
                   Opciones: "text-embedding-3-small" (1536 dims, recomendado),
                            "text-embedding-3-large" (3072 dims),
                            "text-embedding-ada-002" (1536 dims, legacy)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY no está configurada. "
                "Configúrala en las variables de entorno o en el archivo .env"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def embed(self, text: str) -> List[float]:
        """
        Genera un embedding para el texto dado usando la API de OpenAI.
        
        Args:
            text: Texto a convertir en embedding
            
        Returns:
            Lista de floats representando el vector de embedding
            
        Raises:
            ValueError: Si el texto está vacío
            RuntimeError: Si hay un error al llamar a la API
        """
        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Error al generar embedding con OpenAI: {e}")

