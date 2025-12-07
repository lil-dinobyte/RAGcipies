from typing import List, Optional, Callable
import chromadb
from chromadb.config import Settings
from ..models import Chunk
from .base import VectorStore, ScoredChunk


class ChromaDBVectorStore(VectorStore):
    """
    Vector store usando ChromaDB para búsqueda de similitud semántica.
    
    Adecuado para datasets grandes (>10k chunks) y cuando se necesita persistencia.
    ChromaDB maneja automáticamente la indexación y optimización de búsquedas.
    
    Referencias:
    - ChromaDB Docs: https://docs.trychroma.com/
    - ChromaDB Python Client: https://github.com/chroma-core/chroma
    """
    
    def __init__(
        self,
        collection_name: str = "recipes",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Callable] = None
    ):
        """
        Inicializa el ChromaDB vector store.
        
        Args:
            collection_name: Nombre de la colección en ChromaDB
            persist_directory: Directorio para persistencia. Si es None, usa modo en memoria.
                              Si es un string, guarda en disco en ese directorio.
            embedding_function: Función opcional para generar embeddings (no usado en search directo)
        """
        # Configurar cliente de ChromaDB
        if persist_directory:
            # Modo persistente: guarda en disco
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            # Modo en memoria: no persiste datos
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Obtener o crear la colección
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            # Si la colección no existe, crearla
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Usar cosine similarity
            )
        
        self.collection_name = collection_name
        self.embedding_function = embedding_function
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Agrega chunks al vector store de ChromaDB.
        
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
        
        # Convertir Chunk a formato ChromaDB
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        
        # Preparar metadatos: incluir document_id y metadata del chunk
        metadatas = []
        for chunk in chunks:
            metadata = {
                "document_id": chunk.document_id,
                **chunk.metadata  # Incluir metadata adicional del chunk
            }
            metadatas.append(metadata)
        
        # Agregar a ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 3,
        min_score: float = 0.0
    ) -> List[ScoredChunk]:
        """
        Busca los k chunks más similares al query embedding usando ChromaDB.
        
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
        
        # Buscar en ChromaDB
        # ChromaDB retorna distances (menor = más similar)
        # Para cosine similarity, distance = 1 - similarity
        # Entonces: similarity = 1 - distance
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convertir resultados de ChromaDB a ScoredChunk
        scored_chunks = []
        
        if results["ids"] and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for i, (chunk_id, text, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                # Convertir distance a similarity score
                # Para cosine: similarity = 1 - distance
                score = 1.0 - distance
                
                # Filtrar por score mínimo
                if score < min_score:
                    continue
                
                # Extraer document_id del metadata
                document_id = metadata.pop("document_id", chunk_id)
                
                # Reconstruir Chunk
                # Necesitamos el embedding, pero ChromaDB no lo retorna por defecto
                # Para obtenerlo, necesitaríamos hacer otra query o almacenarlo
                # Por ahora, creamos un Chunk sin embedding (se puede mejorar después)
                chunk = Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    text=text,
                    embedding=[],  # ChromaDB no retorna embeddings en query
                    metadata=metadata
                )
                
                # Actualizar metadata con score
                chunk.metadata["similarity_score"] = score
                
                scored_chunks.append(ScoredChunk(chunk=chunk, score=score))
        
        return scored_chunks
    
    def delete(self, ids: List[str]) -> bool:
        """
        Elimina chunks por IDs de ChromaDB.
        
        Args:
            ids: Lista de IDs de chunks a eliminar
            
        Returns:
            True si se eliminaron chunks, False en caso contrario
        """
        if not ids:
            return False
        
        try:
            self.collection.delete(ids=ids)
            return True
        except Exception:
            return False
    
    def __len__(self) -> int:
        """
        Retorna el número de chunks almacenados en ChromaDB.
        
        Returns:
            Número de chunks en la colección
        """
        return self.collection.count()

