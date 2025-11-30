from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RecipeDocument:
    """
    Representa un documento de receta completo.
    
    Attributes:
        id: Identificador único de la receta
        title: Título de la receta
        ingredients: Lista de ingredientes o string con ingredientes
        instructions: Instrucciones de preparación
        tags: Tags opcionales para categorización (ej: ["vegano", "rapido", "sin_gluten"])
    """
    id: str
    title: str
    ingredients: str  # Puede ser string o lista convertida a string
    instructions: str
    tags: Optional[List[str]] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """
        Genera el texto completo de la receta para embedding.
        Incluye título, ingredientes e instrucciones.
        """
        ingredients_text = self.ingredients
        # Si ingredients es una lista en formato string, formatearlo mejor
        if isinstance(self.ingredients, str) and self.ingredients.startswith('['):
            # Intentar parsear si viene como lista serializada
            pass  # Por ahora mantener como está
        
        return (
            f"{self.title}\n\n"
            f"Ingredientes:\n{ingredients_text}\n\n"
            f"Instrucciones:\n{self.instructions}"
        )
    
    def has_tag(self, tag: str) -> bool:
        """
        Verifica si la receta tiene un tag específico.
        
        Args:
            tag: Tag a buscar
            
        Returns:
            True si la receta tiene el tag, False en caso contrario
        """
        return tag.lower() in [t.lower() for t in self.tags]


@dataclass
class Chunk:
    """
    Representa un fragmento (chunk) de texto con su embedding.
    Usado para almacenar y buscar en el vector store.
    
    Attributes:
        id: Identificador único del chunk
        document_id: ID del documento original (RecipeDocument.id)
        text: Texto del chunk
        embedding: Vector de embedding del texto
        metadata: Metadatos opcionales (score de similitud, posición, etc.)
    """
    id: str
    document_id: str
    text: str
    embedding: List[float]
    metadata: Optional[dict] = field(default_factory=dict)
    
    @property
    def similarity_score(self) -> Optional[float]:
        """
        Obtiene el score de similitud del metadata si existe.
        
        Returns:
            Score de similitud o None si no está disponible
        """
        return self.metadata.get("similarity_score")
