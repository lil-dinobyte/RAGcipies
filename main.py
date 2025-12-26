#!/usr/bin/env python3
"""
Ejemplo de uso del pipeline RAG completo.
"""
from dotenv import load_dotenv
load_dotenv()  

from RAGcipies.src.rag.loader import load_recipes_from_json, recipes_to_chunks
from RAGcipies.src.rag.vector_store.factory import (
    create_vector_store,
    VectorStoreBackend
)
from RAGcipies.src.rag.embeddings.factory import EmbeddingBackend
from RAGcipies.src.rag.pipeline import RAGPipeline
from RAGcipies.src.llm.factory import LLMBackend
from pathlib import Path


def main():
    # 1. Cargar recetas
    print("📖 Cargando recetas...")
    recipes_path = Path(__file__).parent / "RAGcipies" / "data" / "recipes.json"
    recipes = load_recipes_from_json(str(recipes_path))
    print(f"✓ {len(recipes)} recetas cargadas")
    
    # 2. Convertir a chunks con embeddings
    print("\n🧮 Generando embeddings...")
    chunks = recipes_to_chunks(recipes, EmbeddingBackend.OLLAMA)
    print(f"✓ {len(chunks)} chunks creados")
    
    # 3. Crear vector store y agregar chunks
    print("\n💾 Creando vector store...")
    chroma_db_path = Path(__file__).parent / "data" / "chroma_db"
    vector_store = create_vector_store(VectorStoreBackend.CHROMADB, persist_directory=str(chroma_db_path))
    vector_store.add_chunks(chunks)
    print(f"✓ Vector store creado con {len(vector_store)} chunks")
    
    # 4. Crear pipeline RAG
    print("\n🔧 Inicializando pipeline RAG...")
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_backend=EmbeddingBackend.OLLAMA,
        llm_backend=LLMBackend.OLLAMA,
        top_k=3
    )
    print("✓ Pipeline RAG listo")
    
    # 5. Hacer algunas consultas
    print("\n" + "=" * 60)
    print("🔍 PROBANDO CONSULTAS")
    print("=" * 60)
    
    queries = [
        "receta con pollo",
        "algo vegano",
        "comida rápida sin horno"
    ]
    
    for query in queries:
        print(f"\n❓ Pregunta: {query}")
        print("-" * 60)
        try:
            response = pipeline.query(query)
            print(f"💬 Respuesta:\n{response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()


if __name__ == "__main__":
    main()