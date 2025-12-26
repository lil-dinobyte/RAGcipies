from typing import Optional
import os
from openai import OpenAI
from .base import LLMClient


class OpenAILLM(LLMClient):
    """
    Cliente LLM usando la API de OpenAI.
    Requiere OPENAI_API_KEY en las variables de entorno.
    
    Referencias:
    - https://platform.openai.com/docs/api-reference/chat
    - https://platform.openai.com/docs/guides/text-generation
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Args:
            model: Modelo de OpenAI a usar.
                   Opciones: "gpt-4o-mini, "gpt-4", "gpt-4-turbo", etc.
            temperature: Controla la aleatoriedad (0.0 = determinista, 1.0 = muy creativo)
            max_tokens: Máximo número de tokens en la respuesta (None = sin límite)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY no está configurada. "
                "Configúrala en las variables de entorno o en el archivo .env"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, prompt: str) -> str:
        """
        Genera una respuesta usando la API de OpenAI.
        
        Args:
            prompt: El prompt completo con contexto y pregunta
            
        Returns:
            La respuesta generada por el LLM
            
        Raises:
            ValueError: Si el prompt está vacío
            RuntimeError: Si hay un error al llamar a la API
        """
        if not prompt or not prompt.strip():
            raise ValueError("El prompt no puede estar vacío")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error al generar respuesta con OpenAI: {e}")