# Adaptado de la Pre-entrega 4 (rag/splitter.py).
from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS

# Separadores priorizando la estructura Markdown: primero secciones, después párrafos y filas de tabla.
SEPARADORES = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]


def split_documents(documentos: List[Document]) -> List[Document]:
    """Fragmenta con RecursiveCharacterTextSplitter midiendo en tokens reales (tiktoken).

    Asigna a cada chunk un id determinístico '<doc_id>-c<n>', así re-ingestar
    sobrescribe los mismos vectores en Pinecone en lugar de duplicarlos.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=SEPARADORES,
    )
    chunks = splitter.split_documents(documentos)

    contador = defaultdict(int)
    for chunk in chunks:
        doc_id = chunk.metadata["doc_id"]
        chunk.metadata["chunk_index"] = contador[doc_id]
        chunk.id = f"{doc_id}-c{contador[doc_id]}"
        contador[doc_id] += 1
    return chunks
