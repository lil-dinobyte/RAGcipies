from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class LLMBase(ABC):
    """
    Clase base abstracta para clientes de LLM.
    Define la interfaz común para generar respuestas.
    """
    
    @abstractmethod
    def generate(
        self, 
        prompt: str,
        **kwargs
    ) -> str:
        """
        Genera una respuesta basada en el prompt dado.
        
        Args:
            prompt: El prompt completo con contexto y pregunta del usuario
            **kwargs: Argumentos adicionales específicos del backend
            
        Returns:
            La respuesta generada por el LLM
            
        Raises:
            ValueError: Si el prompt está vacío
            RuntimeError: Si hay un error al llamar al LLM
        """
        pass
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Permite llamar al LLM como función: llm(prompt)
        
        Args:
            prompt: El prompt a procesar
            **kwargs: Argumentos adicionales
            
        Returns:
            La respuesta generada
        """
        return self.generate(prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre el modelo usado.
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            "backend": self.__class__.__name__,
            "model": getattr(self, "model", "unknown")
        }

LLMClient = LLMBase