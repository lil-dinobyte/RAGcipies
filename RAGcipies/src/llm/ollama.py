from typing import Optional
import os
import requests
from .base import LLMClient


class OllamaLLM(LLMClient):
    """
    Cliente LLM usando Ollama (modelos locales).
    Requiere que Ollama esté corriendo localmente.
    
    Referencias:
    - https://ollama.ai/
    - https://github.com/ollama/ollama
    """
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        timeout: int = 300
    ):
        """
        Args:
            model: Modelo de Ollama a usar (ej: "llama2", "mistral", "codellama")
            base_url: URL base de la API de Ollama
            temperature: Controla la aleatoriedad (0.0 = determinista, 1.0 = muy creativo)
            timeout: Timeout en segundos para las peticiones HTTP (default: 300 = 5 minutos)
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
    
    def generate(self, prompt: str) -> str:
        """
        Genera una respuesta usando la API de Ollama.
        
        Args:
            prompt: El prompt completo con contexto y pregunta
            
        Returns:
            La respuesta generada por el LLM
            
        Raises:
            ValueError: Si el prompt está vacío
            RuntimeError: Si hay un error al llamar a la API o Ollama no está disponible
        """
        if not prompt or not prompt.strip():
            raise ValueError("El prompt no puede estar vacío")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature
                    }
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Error al generar respuesta con Ollama: {e}. "
                f"Asegúrate de que Ollama esté corriendo en {self.base_url}"
            )
        except Exception as e:
            raise RuntimeError(f"Error inesperado al usar Ollama: {e}")