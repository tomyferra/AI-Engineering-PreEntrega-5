"""Tests de la parte local del RAG (carga y chunking): no requieren Pinecone ni API keys."""
from rag.loaders import load_documents
from rag.splitter import split_documents


def test_carga_los_documentos_con_metadata_del_catalogo():
    documentos = {d.metadata["doc_id"]: d for d in load_documents()}
    assert {"historial_ventas_2026", "resultados_financieros_2026", "politicas_comerciales"} <= set(documentos)
    assert documentos["resultados_financieros_2026"].metadata["categoria"] == "finanzas"


def test_chunks_con_ids_deterministicos_y_unicos():
    chunks = split_documents(load_documents())
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [c.id for c in split_documents(load_documents())]
