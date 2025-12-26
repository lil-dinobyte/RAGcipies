from enum import Enum
from typing import Type, Optional
import os
from .base import LLMClient
from .dummy import DummyLLM
from .openai import OpenAILLM
from .ollama import OllamaLLM


class LLMBackend(str, Enum):
    DUMMY = "dummy"
    OPENAI = "openai"
    OLLAMA = "ollama"


# Registry: mapea cada backend a su clase correspondiente (Strategy Pattern)
_LLM_REGISTRY: dict[LLMBackend, Type[LLMClient]] = {
    LLMBackend.DUMMY: DummyLLM,
    LLMBackend.OPENAI: OpenAILLM,
    LLMBackend.OLLAMA: OllamaLLM,
}


def create_llm_client(
    backend: LLMBackend,
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function que crea una instancia del cliente LLM
    según el backend especificado usando el patrón Strategy + Registry.
    
    Args:
        backend: El backend a usar (DUMMY, OPENAI, OLLAMA)
        model: Modelo específico a usar (opcional, usa defaults si no se especifica)
        **kwargs: Argumentos adicionales para el constructor del LLM
                  (temperature, max_tokens, base_url, etc.)
        
    Returns:
        Una instancia del cliente LLM correspondiente
        
    Raises:
        ValueError: Si el backend no está registrado o falta configuración
        
    Ejemplos:
        >>> # Dummy (para pruebas)
        >>> llm = create_llm_client(LLMBackend.DUMMY)
        
        >>> # OpenAI con modelo por defecto
        >>> llm = create_llm_client(LLMBackend.OPENAI)
        
        >>> # OpenAI con modelo específico
        >>> llm = create_llm_client(LLMBackend.OPENAI, model="gpt-4")
        
        >>> # Ollama con configuración personalizada
        >>> llm = create_llm_client(
        ...     LLMBackend.OLLAMA,
        ...     model="mistral",
        ...     base_url="http://localhost:11434"
        ... )
    """
    llm_class = _LLM_REGISTRY.get(backend)
    
    if llm_class is None:
        available = ", ".join(b.value for b in _LLM_REGISTRY.keys())
        raise ValueError(
            f"Invalid LLM backend: {backend.value}. "
            f"Available backends: {available}"
        )
    
    # Si no se especifica model, intentar leerlo de variables de entorno
    if model is None:
        if backend == LLMBackend.OPENAI:
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        elif backend == LLMBackend.OLLAMA:
            model = os.getenv("OLLAMA_MODEL", "llama2")
    
    # Construir argumentos para el constructor
    init_kwargs = {"model": model} if model else {}
    init_kwargs.update(kwargs)
    
    # Para Ollama, leer base_url de env si no se especifica
    if backend == LLMBackend.OLLAMA and "base_url" not in init_kwargs:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        init_kwargs["base_url"] = base_url
    
    return llm_class(**init_kwargs)