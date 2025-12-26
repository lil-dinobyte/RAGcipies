import json
from pathlib import Path
from typing import List
from .models import RecipeDocument, Chunk
from .embeddings.factory import create_embedding_model, EmbeddingBackend


def load_recipes_from_json(json_path: str) -> List[RecipeDocument]:
    """
    Carga recetas desde un archivo JSON.
    
    Args:
        json_path: Ruta al archivo JSON con recetas
        
    Returns:
        Lista de RecipeDocument
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el JSON es inválido
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {json_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    recipes = []
    for item in data:
        # Convertir ingredients de lista a string si es necesario
        ingredients = item.get("ingredients", [])
        if isinstance(ingredients, list):
            ingredients = ", ".join(ingredients)
        
        recipe = RecipeDocument(
            id=item.get("id", ""),
            title=item.get("title", ""),
            ingredients=ingredients,
            instructions=item.get("instructions", ""),
            tags=item.get("tags", [])
        )
        recipes.append(recipe)
    
    return recipes


def recipes_to_chunks(
    recipes: List[RecipeDocument],
    embedding_backend: EmbeddingBackend = EmbeddingBackend.FAKE
) -> List[Chunk]:
    """
    Convierte recetas en Chunks con embeddings.
    
    Args:
        recipes: Lista de recetas a convertir
        embedding_backend: Backend de embeddings a usar
        
    Returns:
        Lista de Chunks con embeddings calculados
    """
    embedding_model = create_embedding_model(embedding_backend)
    chunks = []
    
    for recipe in recipes:
        # Usar el texto completo de la receta
        text = recipe.full_text
        
        # Calcular embedding
        embedding = embedding_model.embed(text)
        
        # Crear chunk
        chunk = Chunk(
            id=f"chunk_{recipe.id}",
            document_id=recipe.id,
            text=text,
            embedding=embedding,
            metadata={
                "title": recipe.title,
                "tags": recipe.tags,
                "source": "recipes.json"
            }
        )
        chunks.append(chunk)
    
    return chunks