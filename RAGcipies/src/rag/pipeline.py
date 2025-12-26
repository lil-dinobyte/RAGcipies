from typing import Optional
from .embeddings.factory import create_embedding_model, EmbeddingBackend
from .vector_store.factory import create_vector_store, VectorStoreBackend
from .vector_store.base import VectorStore
from ..llm.prompt.builder import PromptBuilder
from ..llm.factory import create_llm_client, LLMBackend


class RAGPipeline:
    """
    Pipeline RAG completo que orquesta:
    1. Embedding de la query
    2. Búsqueda vectorial
    3. Construcción del prompt
    4. Generación con LLM
    5. Retorno de la respuesta
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_backend: EmbeddingBackend = EmbeddingBackend.FAKE,
        llm_backend: LLMBackend = LLMBackend.DUMMY,
        top_k: int = 3,
        min_score: float = 0.0,
        include_scores_in_prompt: bool = False,
        prompt_template_path: Optional[str] = None
    ):
        """
        Inicializa el pipeline RAG.
        
        Args:
            vector_store: Instancia del vector store (ya inicializado con datos)
            embedding_backend: Backend de embeddings a usar
            llm_backend: Backend de LLM a usar
            top_k: Número de chunks a recuperar en la búsqueda
            min_score: Score mínimo de similitud para filtrar resultados
            include_scores_in_prompt: Si True, incluye scores en el prompt
            prompt_template_path: Ruta al template YAML (opcional)
        """
        self.vector_store = vector_store
        self.embedding_model = create_embedding_model(embedding_backend)
        self.llm = create_llm_client(llm_backend)
        self.top_k = top_k
        self.min_score = min_score
        self.prompt_builder = PromptBuilder(template_path=prompt_template_path)
        self.include_scores_in_prompt = include_scores_in_prompt
    
    def query(self, user_query: str) -> str:
        """
        Ejecuta el pipeline RAG completo para una consulta del usuario.
        
        Args:
            user_query: La pregunta del usuario en lenguaje natural
            
        Returns:
            La respuesta generada por el LLM basada en el contexto recuperado
            
        Raises:
            ValueError: Si la query está vacía
            RuntimeError: Si hay un error en algún paso del pipeline
        """
        if not user_query or not user_query.strip():
            raise ValueError("La consulta del usuario no puede estar vacía")
        
        # Paso 1: Convertir query a embedding
        query_embedding = self.embedding_model.embed(user_query)
        
        # Paso 2: Buscar chunks similares en el vector store
        scored_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            k=self.top_k,
            min_score=self.min_score
        )
        
        # Paso 3: Construir el prompt con contexto
        prompt = self.prompt_builder.build(
            query=user_query,
            scored_chunks=scored_chunks,
            include_scores=self.include_scores_in_prompt
        )
        
        # Paso 4: Generar respuesta con el LLM
        response = self.llm.generate(prompt)
        
        # Paso 5: Retornar la respuesta
        return response