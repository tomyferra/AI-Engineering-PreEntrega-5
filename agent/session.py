"""Armado del agente completo (LLM + tools + checkpointer) y ejecución de un turno de conversación."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from agent.checkpointer import open_checkpointer
from agent.graph import build_graph
from agent.llm import build_chat_model
from agent.trace import TraceRecorder
from config import CHECKPOINTS_DB_PATH, PEDIDOS_DB_PATH, RECURSION_LIMIT, get_llm_settings, get_rag_settings
from db.repository import PedidosRepository
from rag.retriever import RAGSystem
from tools import build_tools

logger = logging.getLogger(__name__)


@asynccontextmanager
async def create_agent(use_rag: bool = True, checkpoints_path: Path = CHECKPOINTS_DB_PATH,
                       pedidos_path: Path = PEDIDOS_DB_PATH) -> AsyncIterator[CompiledStateGraph]:
    """Construye el agente listo para usar y cierra el checkpointer al salir (llamar después de load_dotenv)."""
    model = build_chat_model(get_llm_settings())
    rag_factory = None
    if use_rag:
        rag_settings = get_rag_settings()  # valida las variables de Pinecone antes de arrancar
        rag_factory = lambda: RAGSystem(rag_settings)  # noqa: E731 - se instancia recién en la 1ª consulta
    tools = build_tools(PedidosRepository(pedidos_path), rag_factory)

    async with open_checkpointer(checkpoints_path) as checkpointer:
        yield build_graph(model, tools, checkpointer=checkpointer)


def run_config(thread_id: str, recursion_limit: int = RECURSION_LIMIT) -> dict:
    """thread_id identifica la sesión en el checkpointer; recursion_limit pone techo a los pasos del grafo."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


async def run_turn(graph: CompiledStateGraph, thread_id: str, mensaje: str,
                   recorder: Optional[TraceRecorder] = None, recursion_limit: int = RECURSION_LIMIT) -> str:
    """Envía un mensaje al agente y devuelve la respuesta final, registrando cada paso del ciclo ReAct.

    Solo se envía el mensaje nuevo: el historial previo del thread lo recupera el checkpointer.
    """
    recorder = recorder or TraceRecorder(echo=None)
    recorder.start_turn(thread_id, mensaje)
    respuesta: Optional[str] = None
    try:
        async for update in graph.astream({"messages": [HumanMessage(mensaje)]},
                                          config=run_config(thread_id, recursion_limit), stream_mode="updates"):
            for nodo, cambios in update.items():
                mensajes = (cambios or {}).get("messages", [])
                recorder.on_update(nodo, mensajes)
                for m in mensajes:
                    if isinstance(m, AIMessage) and not m.tool_calls:
                        respuesta = m.text
    except GraphRecursionError:
        error = f"Se alcanzó el límite de {recursion_limit} pasos sin una respuesta final."
        logger.warning(error)
        recorder.end_turn(None, error=error)
        return f"No pude completar la consulta: {error} Probá con una pregunta más acotada."
    recorder.end_turn(respuesta)
    return respuesta or ""
