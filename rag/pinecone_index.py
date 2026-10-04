# Copiado de la Pre-entrega 4 (rag/pinecone_index.py), adaptado a RAGSettings.
import logging
import time

from pinecone import Pinecone, ServerlessSpec
from pinecone.db_data.index import Index

from config import RAGSettings

logger = logging.getLogger(__name__)


def get_pinecone_client(settings: RAGSettings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key.get_secret_value())


def ensure_index(settings: RAGSettings, timeout_s: int = 120) -> Index:
    """Verifica si el índice existe y lo crea en modo Serverless si hace falta.

    Si ya existe, valida que su dimensión coincida con la de los embeddings para evitar
    el clásico mismatch (p. ej. subir vectores de 1536 a un índice de 768).
    """
    pc = get_pinecone_client(settings)

    if pc.has_index(settings.index_name):
        descripcion = pc.describe_index(settings.index_name)
        if descripcion.dimension != settings.dimension:
            raise ValueError(
                f"El índice '{settings.index_name}' tiene dimensión {descripcion.dimension}, "
                f"pero los embeddings configurados generan {settings.dimension}. "
                f"Usá otro INDEX_NAME o borrá el índice existente."
            )
        logger.info("El índice '%s' ya existe (dim=%d, metric=%s).",
                    settings.index_name, descripcion.dimension, descripcion.metric)
    else:
        logger.info("Creando índice Serverless '%s' (dim=%d, %s/%s)...",
                    settings.index_name, settings.dimension, settings.cloud, settings.region)
        pc.create_index(
            name=settings.index_name,
            dimension=settings.dimension,
            metric=settings.metric,
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
        )

    inicio = time.monotonic()
    while not pc.describe_index(settings.index_name).status["ready"]:
        if time.monotonic() - inicio > timeout_s:
            raise TimeoutError(f"El índice '{settings.index_name}' no quedó listo en {timeout_s}s")
        time.sleep(2)

    return pc.Index(settings.index_name)
