# Copiado de la Pre-entrega 4 (rag/retriever.py), adaptado a RAGSettings.
import re
from typing import List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import CANDIDATES_K, ENSEMBLE_WEIGHTS, TOP_K, RAGSettings
from rag.ingestion import get_vectorstore
from rag.loaders import load_documents
from rag.pinecone_index import ensure_index
from rag.splitter import split_documents


def tokenize(texto: str) -> List[str]:
    """Tokenizador para BM25: minúsculas y palabras alfanuméricas (conserva 'q2', 'x50', 'gp-20' -> 'gp','20')."""
    return re.findall(r"\w+", texto.lower())


class RAGSystem:
    """Recuperador híbrido: combina BM25 (léxico, local) y Pinecone (semántico, en la nube).

    BM25 es clave en este corpus: términos exactos como "Q2", "x50" o "septiembre" aparecen en
    tablas que un embedding no siempre distingue bien. Ambos se fusionan con Reciprocal Rank Fusion.
    """

    def __init__(self, settings: RAGSettings, top_k: int = TOP_K, candidates_k: int = CANDIDATES_K,
                 weights: List[float] = ENSEMBLE_WEIGHTS):
        self.top_k = top_k
        index = ensure_index(settings)
        vector_retriever = get_vectorstore(settings, index).as_retriever(search_kwargs={"k": candidates_k})
        # BM25 necesita el corpus en memoria: se reconstruye con el mismo loader + splitter
        # de la ingesta, así los chunks (y sus ids) coinciden con los de Pinecone.
        bm25_retriever = BM25Retriever.from_documents(
            split_documents(load_documents()), k=candidates_k, preprocess_func=tokenize
        )
        self.ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=weights)

    def retrieve(self, query: str) -> List[Document]:
        return self.ensemble.invoke(query)[: self.top_k]
