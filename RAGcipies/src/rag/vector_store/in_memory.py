from typing import List, Tuple
import math
from ..models import Chunk
from .base import VectorStore, ScoredChunk


class InMemoryVectorStore(VectorStore):
    """
    Vector store en memoria para búsqueda de similitud semántica.
    Usa cosine similarity para encontrar los chunks más similares.
    
    Adecuado para datasets pequeños/medianos (<10k chunks).
    No requiere dependencias externas más allá de Python estándar.
    
    Referencias:
    - Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
    - Vector Search: https://www.pinecone.io/learn/vector-search/
    """
    
    def __init__(self) -> None:
        """Inicializa un vector store vacío."""
        self._chunks: List[Chunk] = []
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Agrega chunks al vector store.
        
        Args:
            chunks: Lista de chunks a agregar
            
        Raises:
            ValueError: Si la lista está vacía o algún chunk no tiene embedding
        """
        if not chunks:
            raise ValueError("No se pueden agregar chunks vacíos")
        
        # Validar que todos los chunks tengan embeddings
        for chunk in chunks:
            if not chunk.embedding:
                raise ValueError(f"Chunk {chunk.id} no tiene embedding")
        
        self._chunks.extend(chunks)
    
    def add_chunk(self, chunk: Chunk) -> None:
        """
        Agrega un solo chunk al vector store.
        
        Args:
            chunk: Chunk a agregar
            
        Raises:
            ValueError: Si el chunk no tiene embedding
        """
        if not chunk.embedding:
            raise ValueError(f"Chunk {chunk.id} no tiene embedding")
        self._chunks.append(chunk)
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        Calcula la similitud del coseno entre dos vectores.
        
        La similitud del coseno mide el ángulo entre dos vectores:
        - 1.0 = vectores idénticos (mismo ángulo)
        - 0.0 = vectores ortogonales (90 grados)
        - -1.0 = vectores opuestos (180 grados)
        
        Args:
            a: Primer vector
            b: Segundo vector
            
        Returns:
            Score de similitud entre -1.0 y 1.0
            
        Raises:
            ValueError: Si los vectores tienen dimensiones diferentes o están vacíos
        """
        if len(a) != len(b):
            raise ValueError(
                f"Vectores deben tener la misma dimensión: {len(a)} != {len(b)}"
            )
        
        if not a or not b:
            raise ValueError("Vectores no pueden estar vacíos")
        
        # Producto punto
        dot_product = sum(x * y for x, y in zip(a, b))
        
        # Normas (magnitudes)
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        # Evitar división por cero
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 3,
        min_score: float = 0.0
    ) -> List[ScoredChunk]:
        """
        Busca los k chunks más similares al query embedding.
        
        Args:
            query_embedding: Vector de embedding de la consulta
            k: Número de resultados a retornar (top-k)
            min_score: Score mínimo de similitud (filtra resultados con score menor)
            
        Returns:
            Lista de ScoredChunk ordenados por score descendente
            
        Raises:
            ValueError: Si query_embedding está vacío o k es inválido
        """
        if not query_embedding:
            raise ValueError("query_embedding no puede estar vacío")
        
        if k <= 0:
            raise ValueError(f"k debe ser mayor a 0, recibido: {k}")
        
        if not self._chunks:
            return []  # Retornar lista vacía si no hay chunks
        
        # Calcular similitud para todos los chunks
        scored: List[Tuple[Chunk, float]] = []
        for chunk in self._chunks:
            try:
                score = self._cosine_similarity(query_embedding, chunk.embedding)
                
                # Filtrar por score mínimo
                if score >= min_score:
                    scored.append((chunk, score))
            except ValueError as e:
                # Si hay error de dimensión, saltar este chunk
                # (podría loguear el error en producción)
                continue
        
        # Ordenar por score descendente
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Limitar a k resultados (o menos si hay menos chunks)
        k = min(k, len(scored))
        
        # Crear ScoredChunk y actualizar metadata
        results = []
        for chunk, score in scored[:k]:
            # Actualizar metadata del chunk con el score
            chunk.metadata["similarity_score"] = score
            results.append(ScoredChunk(chunk=chunk, score=score))
        
        return results
    
    def delete(self, ids: List[str]) -> bool:
        """
        Elimina chunks por IDs.
        
        Args:
            ids: Lista de IDs de chunks a eliminar
            
        Returns:
            True si se eliminaron chunks, False en caso contrario
        """
        initial_count = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.id not in ids]
        return len(self._chunks) < initial_count
    
    def clear(self) -> None:
        """Limpia todos los chunks del vector store."""
        self._chunks.clear()
    
    def __len__(self) -> int:
        """Retorna el número de chunks almacenados."""
        return len(self._chunks)

