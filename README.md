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

y esa receta solo existe en nuestro recetario, un LLM sin RAG:

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

## 🔁 Flujo de trabajo

```
(1) Pregunta del usuario
        │
        ▼
(2) Embedding de la query (EmbeddingModel)
        │
        ▼
(3) Búsqueda vectorial (VectorStore.search → top-k chunks)
        │
        ▼
(4) Armado de contexto + prompt (build_prompt)
        │
        ▼
(5) LLM (LLMClient.generate)
        │
        ▼
(6) Respuesta final al usuario

```

### 🔍 1. Pregunta del usuario

El usuario envía una consulta en lenguaje natural.
Ejemplo:

> “Quiero una receta con pollo y arroz”

Este texto es la entrada principal al flujo RAG.

### 🧮 2. Embedding de la query (EmbeddingModel)

La pregunta del usuario se convierte en un **vector numérico** mediante un modelo de embeddings. 

**¿Qué es un embedding?**
Un embedding es una representación matemática del significado de un texto. Transforma palabras y frases en vectores (listas de números) que capturan la semántica del contenido.

Ejemplo:

```
"Pollo al curry" → [0.02, -0.11, 0.79, 0.45, -0.23, 0.67, ...]
"Pollo con curry" → [0.03, -0.10, 0.78, 0.44, -0.22, 0.66, ...]  (similar)
"Ensalada de frutas" → [0.91, 0.23, -0.45, 0.12, 0.88, -0.34, ...]  (diferente)
```

**¿Cómo funciona?**
Los embeddings operan en un "espacio semántico" donde:
- **Textos similares** (como "pollo al curry" y "pollo con curry") tienen vectores **cercanos** entre sí.
- **Textos diferentes** (como "pollo al curry" y "ensalada de frutas") tienen vectores **lejos** entre sí.

Esta propiedad permite buscar información por **similitud semántica** en lugar de solo coincidencias exactas de palabras.

**En el flujo RAG:**
El embedding de la consulta se usa para encontrar las recetas más relevantes en el vector store mediante búsqueda por similitud (similarity search), que es el siguiente paso del pipeline.

### 🧭 3. Búsqueda vectorial (VectorStore.search)

El vector de la query se compara con los embeddings de todas las recetas almacenadas.
El vector store calcula la similitud (por ejemplo, cosine similarity) y devuelve los **top-k** documentos más relevantes.

```
1. Pollo con arroz (score 0.95)
2. Arroz con pollo al horno (score 0.88)
3. Arroz con verduras (score 0.61)
```

### 🧱 4. Armado del contexto + prompt (build_prompt)

El sistema arma un prompt que incluye:
- instrucciones para el modelo,
- los chunks/documentos recuperados,
- la pregunta original del usuario.

```
Usa el siguiente contexto para responder la pregunta:

[score=0.95]
Receta: Pollo con arroz...
Ingredientes...
Instrucciones...

Pregunta del usuario: "Quiero una receta con pollo y arroz"

```

### 🤖 5. LLM (LLMClient.generate)

El prompt se envía al modelo de lenguaje (LLM).
El LLM genera una respuesta utilizando:

- el contexto recuperado,
- su conocimiento general,
- y la pregunta del usuario.

Si se usa `DummyLLM`, la respuesta es un placeholder para pruebas.
Cuando se integre un LLM real, producirá recetas completas y relevantes.

### 📝 6. Respuesta final al usuario

El pipeline retorna la respuesta del LLM, por ejemplo:

> "Podés preparar un pollo con arroz salteado con ajo y cebolla.
> Aquí tenés una receta basada en el contexto recuperado…"

Este es el resultado final para mostrar en una API, notebook o interfaz.














## 🔗 Links

### RAG (Retrieval-Augmented Generation)

- [IBM - Retrieval-Augmented Generation](https://www.ibm.com/think/topics/retrieval-augmented-generation)
- [Google Cloud - RAG Use Cases](https://cloud.google.com/use-cases/retrieval-augmented-generation)
- [NVIDIA Blog - What is RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)

### Embeddings

#### OpenAI Embeddings (Documentación Oficial)

- [OpenAI API Reference - Embeddings](https://platform.openai.com/docs/api-reference/embeddings) - Especificación completa de la API, parámetros y modelos disponibles
- [OpenAI Python SDK](https://github.com/openai/openai-python) - Repositorio oficial del SDK de Python
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) - Guía práctica de uso y mejores prácticas
- [OpenAI Models - Embeddings](https://platform.openai.com/docs/models/embeddings) - Modelos de embeddings disponibles (text-embedding-3-small, text-embedding-3-large, etc.)
- [OpenAI Cookbook - Embeddings](https://cookbook.openai.com/examples/how_to_get_embeddings) - Ejemplos prácticos y casos de uso
- [OpenAI Cookbook Repository](https://github.com/openai/openai-cookbook) - Repositorio con ejemplos, notebooks y tutoriales