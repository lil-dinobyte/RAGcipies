from typing import List, Optional
from pathlib import Path
import yaml
from RAGcipies.src.rag.vector_store import ScoredChunk


class PromptBuilder:
    """
    Constructor de prompts que lee templates desde archivos YAML.
    Permite personalizar el formato del prompt sin modificar código.
    """
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Inicializa el PromptBuilder.
        
        Args:
            template_path: Ruta al archivo YAML con el template.
                          Si es None, usa el template por defecto.
        """
        if template_path is None:
            # Ruta por defecto: mismo directorio que este archivo
            default_path = Path(__file__).parent / "prompt.yaml"
            template_path = str(default_path)
        
        self.template_path = Path(template_path)
        self._load_template()
    
    def _load_template(self) -> None:
        """Carga el template desde el archivo YAML."""
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Template no encontrado: {self.template_path}. "
                "Asegúrate de que el archivo YAML existe."
            )
        
        with open(self.template_path, "r", encoding="utf-8") as f:
            self.template = yaml.safe_load(f)
    
    def build(
        self,
        query: str,
        scored_chunks: List[ScoredChunk],
        include_scores: Optional[bool] = None
    ) -> str:
        """
        Construye un prompt usando el template cargado.
        
        Args:
            query: La pregunta original del usuario
            scored_chunks: Lista de chunks recuperados con sus scores
            include_scores: Si True, incluye scores. Si None, usa el valor del template
            
        Returns:
            El prompt completo formateado
        """
        if include_scores is None:
            include_scores = self.template.get("include_scores", False)
        
        # Construir el contexto
        if not scored_chunks:
            context_section = self.template.get("no_context_message", "")
        else:
            context_parts = []
            for i, scored_chunk in enumerate(scored_chunks, 1):
                chunk = scored_chunk.chunk
                score = scored_chunk.score
                
                chunk_text = f"Receta {i}:\n{chunk.text}"
                
                if include_scores:
                    chunk_text = f"[Relevancia: {score:.2f}]\n{chunk_text}"
                
                context_parts.append(chunk_text)
            
            separator = self.template.get("chunk_separator", "\n\n---\n\n")
            context = separator.join(context_parts)
            
            context_header = self.template.get("context_header", "CONTEXTO:")
            context_section = f"{context_header}\n\n{context}"
        
        # Construir el prompt completo
        system_instruction = self.template.get("system_instruction", "")
        instructions = self.template.get("instructions", "")
        instructions_header = self.template.get("instructions_header", "INSTRUCCIONES:")
        question_header = self.template.get("question_header", "PREGUNTA DEL USUARIO:")
        response_header = self.template.get("response_header", "RESPUESTA:")
        
        prompt_parts = [
            system_instruction,
            "",
            context_section,
            "",
            f"{instructions_header}",
            instructions,
            "",
            f"{question_header}",
            query,
            "",
            f"{response_header}"
        ]
        
        return "\n".join(prompt_parts)


# Función de conveniencia (backward compatibility)
def build_prompt(
    query: str,
    scored_chunks: List[ScoredChunk],
    include_scores: bool = False,
    template_path: Optional[str] = None
) -> str:
    """
    Función de conveniencia para construir un prompt.
    
    Args:
        query: La pregunta del usuario
        scored_chunks: Chunks recuperados
        include_scores: Si incluir scores en el prompt
        template_path: Ruta al template YAML (opcional)
        
    Returns:
        El prompt formateado
    """
    builder = PromptBuilder(template_path=template_path)
    return builder.build(query, scored_chunks, include_scores)