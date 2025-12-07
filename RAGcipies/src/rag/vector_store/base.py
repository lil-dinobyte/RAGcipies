from abc import ABC, abstractmethod
from typing import List
from ..models import Chunk


class ScoredChunk:
    """
    Representa un Chunk con su score de similitud.
    
    Attributes:
        chunk: El chunk encontrado
        score: Score de similitud (0.0 a 1.0 para cosine similarity)
    """
    def __init__(self, chunk: Chunk, score: float):
        self.chunk = chunk
        self.score = score
    
    def __repr__(self) -> str:
        return f"ScoredChunk(chunk_id={self.chunk.id}, score={self.score:.4f})"


class VectorStore(ABC):
    """
    Clase base abstracta para implementaciones de VectorStore.
    Define la interfaz común para almacenar y buscar chunks con embeddings.
    """
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Agrega chunks al vector store.
        
        Args:
            chunks: Lista de chunks a agregar
        """
        pass
    
    @abstractmethod
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
        """
        pass
    
    def delete(self, ids: List[str]) -> bool:
        """
        Elimina chunks por IDs.
        
        Args:
            ids: Lista de IDs de chunks a eliminar
            
        Returns:
            True si se eliminaron chunks, False en caso contrario
            
        Note:
            Implementación opcional. Si no se implementa, retorna False.
        """
        return False
    
    def __len__(self) -> int:
        """
        Retorna el número de chunks almacenados.
        
        Returns:
            Número de chunks en el store
            
        Note:
            Implementación opcional. Si no se implementa, retorna 0.
        """
        return 0

