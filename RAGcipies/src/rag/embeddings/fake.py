import hashlib
import math
from typing import List
from .base import EmbeddingModel


class FakeEmbeddingModel(EmbeddingModel):
    """
    Embedding determinista y liviano.
    No usa APIs externas y siempre da el mismo resultado
    para el mismo texto.
    """

    def embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()

        # Vector chico (8 dimensiones)
        vector = [b / 255.0 for b in digest[:8]]

        # Normalizamos
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
