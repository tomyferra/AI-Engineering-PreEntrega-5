"""Tool de RAG: consulta la base de conocimiento de GenPet (Pinecone + BM25) indexada con ingest.py."""
import asyncio
import logging
from typing import Callable, Optional

from langchain_core.tools import BaseTool, tool

from rag.retriever import RAGSystem
from schemas import BuscarInformacionInput

logger = logging.getLogger(__name__)


def build_knowledge_tool(rag_factory: Callable[[], RAGSystem]) -> BaseTool:
    """Crea la tool con inicialización perezosa: el RAGSystem (conexión a Pinecone + corpus BM25 en memoria)
    se construye recién la primera vez que el agente lo necesita, y se reutiliza en las siguientes llamadas."""
    rag: Optional[RAGSystem] = None
    lock = asyncio.Lock()

    async def _get_rag() -> RAGSystem:
        nonlocal rag
        async with lock:
            if rag is None:
                rag = await asyncio.to_thread(rag_factory)
        return rag

    @tool(args_schema=BuscarInformacionInput)
    async def buscar_informacion_empresa(consulta: str) -> dict:
        """Busca en los documentos internos de GenPet y devuelve los fragmentos más relevantes con su fuente.
        Cubre: historial de ventas 2026 por mes, trimestre, producto, canal y top clientes; resultados financieros
        (ingresos, costos, margen y ganancia neta por mes y por trimestre); plan comercial y metas 2026;
        políticas comerciales (descuentos por tipo de cliente, envíos, medios de pago, devoluciones); ficha técnica
        de los paños 60x90 cm; e historia, equipo y operación de la empresa.
        NO sirve para pedidos de un cliente puntual (para eso usar las tools de pedidos).
        Hacé consultas específicas; si la pregunta tiene varias partes (por ejemplo ganancias Y ventas por producto),
        llamala una vez por cada parte."""
        try:
            sistema = await _get_rag()
            documentos = await asyncio.to_thread(sistema.retrieve, consulta)
        except Exception as exc:  # Pinecone caído, API key inválida, timeout...
            logger.exception("Falló la búsqueda en la base de conocimiento")
            return {"error": f"La base de conocimiento no está disponible: {type(exc).__name__}: {exc}",
                    "accion_sugerida": "Informale al usuario que no se pudo consultar la información interna."}

        if not documentos:
            return {"error": "No se encontraron fragmentos relevantes.",
                    "accion_sugerida": "Reformulá la consulta con otras palabras o pedile al usuario más detalle."}
        return {"fragmentos": [{"fuente": d.metadata.get("source"), "categoria": d.metadata.get("categoria"),
                                "contenido": d.page_content} for d in documentos]}

    return buscar_informacion_empresa
