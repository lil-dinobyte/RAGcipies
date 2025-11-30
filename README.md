# RAGcipies 🥣

RAGcipies es un sistema de Retrieval-Augmented Generation (RAG) que utiliza búsqueda vectorial semántica para encontrar recetas de cocina basándose en consultas en lenguaje natural como:

- "cena vegana con garbanzos"
- "algo dulce sin huevos"
- "comida rápida con arroz"

El sistema almacena una colección de recetas, las convierte en embeddings, las inserta en una base de datos vectorial, recupera los resultados más relevantes semánticamente, y usa un LLM para generar una respuesta útil basada en el contexto recuperado.

## 🎯 Objetivos Principales

- Aprender e implementar un pipeline RAG minimalista y limpio
- Usar búsqueda vectorial para encontrar recetas relevantes
- Usar un LLM para formatear y generar la respuesta final

## 🧠 ¿Qué es RAG y qué problema resuelve?

**RAG (Retrieval-Augmented Generation)** es un patrón de arquitectura para LLMs que combina:

1. **Recuperación de información** desde una base de conocimiento externa (vector store, base de datos, archivos, etc.).
2. **Generación con un modelo de lenguaje (LLM)** usando esa información como contexto.

La idea central es:

> El LLM **no inventa** la respuesta, sino que **lee primero** desde una fuente confiable y después genera la respuesta usando ese contexto.

### ¿Por qué es útil en un proyecto de recetas?

Un LLM “puro” solo conoce recetas que vio durante su entrenamiento.  
No sabe nada sobre:

- Recetas nuevas creadas por el usuario.
- Recetas privadas almacenadas en un JSON local.
- Recetas con formatos o combinaciones específicas que no existen en Internet.

Si le pedís, por ejemplo:

> "tarta de pollo con masa de avena y curry"

y esa receta solo existe en `recipes.json`, un LLM sin RAG:

- puede **alucinar** (inventar una receta parecida),
- o puede devolver algo genérico que no coincide con tu dataset.

Con RAG, el flujo cambia:

1. La pregunta del usuario se convierte en un embedding.
2. Se busca en el **vector store** de recetas las más parecidas semánticamente.
3. Se recuperan las recetas relevantes.
4. El LLM genera la respuesta **basado en esas recetas reales**.

De esta forma:

- RAGcipies puede trabajar con **recetas que no existían cuando se entrenó el modelo**.
- Se soportan **recetas privadas y personalizadas**.
- Se reducen las **alucinaciones**, porque el modelo se apoya en evidencia real (las recetas del dataset).