# Copiado de la Pre-entrega 4 (rag/ingestion.py), adaptado a RAGSettings.
import logging

from langchain_pinecone import PineconeVectorStore
from pinecone.db_data.index import Index
from pinecone.exceptions import NotFoundException

from config import RAGSettings
from rag.embeddings import get_embeddings
from rag.loaders import load_documents
from rag.pinecone_index import ensure_index
from rag.splitter import split_documents

logger = logging.getLogger(__name__)

# PineconeVectorStore guarda el texto original del chunk en esta clave de la metadata,
# así la recuperación no necesita consultar otra base de datos.
TEXT_KEY = "text"


def count_vectors(index: Index, namespace: str) -> int:
    stats = index.describe_index_stats()
    ns = stats.namespaces.get(namespace)
    return ns.vector_count if ns else 0


def get_vectorstore(settings: RAGSettings, index: Index) -> PineconeVectorStore:
    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(settings),
        namespace=settings.namespace,
        text_key=TEXT_KEY,
    )


def ingest(settings: RAGSettings, force: bool = False) -> int:
    """Carga, fragmenta, genera embeddings y sube los chunks a Pinecone. Devuelve la cantidad subida.

    Si el namespace ya tiene vectores no reindexa, salvo force=True (que primero lo vacía).
    """
    index = ensure_index(settings)
    existentes = count_vectors(index, settings.namespace)

    if existentes and not force:
        logger.info("El namespace '%s' ya tiene %d vectores; se omite la ingesta (usá --force para reindexar).",
                    settings.namespace, existentes)
        return 0

    if existentes and force:
        logger.info("Vaciando namespace '%s' (%d vectores)...", settings.namespace, existentes)
        try:
            index.delete(delete_all=True, namespace=settings.namespace)
        except NotFoundException:
            pass

    documentos = load_documents()
    chunks = split_documents(documentos)
    logger.info("Subiendo %d chunks de %d documentos al namespace '%s' con embeddings %s...",
                len(chunks), len(documentos), settings.namespace, settings.embedding_model)

    vectorstore = get_vectorstore(settings, index)
    vectorstore.add_documents(chunks, ids=[c.id for c in chunks])
    logger.info("Ingesta completada: %d chunks en '%s/%s'.", len(chunks), settings.index_name, settings.namespace)
    return len(chunks)
