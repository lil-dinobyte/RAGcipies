from .base import LLMClient


class DummyLLM(LLMClient):
    """
    LLM dummy para pruebas y desarrollo.
    No usa APIs externas y siempre retorna una respuesta predefinida.
    Útil para probar el pipeline RAG sin consumir tokens.
    """
    
    def generate(self, prompt: str) -> str:
        """
        Genera una respuesta dummy basada en el prompt.
        
        Args:
            prompt: El prompt completo
            
        Returns:
            Una respuesta placeholder
        """
        if not prompt or not prompt.strip():
            raise ValueError("El prompt no puede estar vacío")
        
        # Respuesta dummy que indica que recibió el prompt
        return (
            "[Dummy LLM Response]\n\n"
            "Este es un placeholder. El prompt recibido fue:\n\n"
            f"{prompt[:200]}...\n\n"
            "(En producción, aquí aparecería la respuesta real del LLM)"
        )