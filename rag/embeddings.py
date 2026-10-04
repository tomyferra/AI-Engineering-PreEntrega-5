# Adaptado de la Pre-entrega 4 (rag/embeddings.py).
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from config import EmbeddingProvider, RAGSettings


def get_embeddings(settings: RAGSettings) -> Embeddings:
    """Instancia el modelo de embeddings del provider configurado, siempre con la dimensión del índice."""
    api_key = settings.get_embedding_api_key()
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddings(api_key=api_key, model=settings.embedding_model, dimensions=settings.dimension)
    if settings.embedding_provider == EmbeddingProvider.GEMINI:
        return GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model=settings.embedding_model,
            output_dimensionality=settings.dimension,
        )
    raise ValueError(f"Embedding provider no soportado: {settings.embedding_provider}")
