"""Turn chunk text into vectors with MiniLM.

The model is loaded once per process and reused. Vectors are length 384.
They are normalized so later search can use cosine distance.
"""

from app.config import get_settings

VECTOR_SIZE = 384

# Held on the process so the second document does not load the model again.
_embedding_model = None


def load_embedding_model():
    """Load `sentence-transformers/all-MiniLM-L6-v2` the first time we need it."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(get_settings().embedding_model)
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one 384-number vector for each text."""
    model = load_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]
