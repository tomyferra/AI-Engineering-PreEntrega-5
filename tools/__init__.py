from typing import Callable, List, Optional

from langchain_core.tools import BaseTool

from db.repository import PedidosRepository
from rag.retriever import RAGSystem
from tools.conocimiento import build_knowledge_tool
from tools.pedidos import build_pedidos_tools


def build_tools(repo: PedidosRepository, rag_factory: Optional[Callable[[], RAGSystem]] = None) -> List[BaseTool]:
    """Todas las tools del agente. Sin rag_factory, el agente funciona solo con las tools de pedidos."""
    tools = build_pedidos_tools(repo)
    if rag_factory is not None:
        tools.append(build_knowledge_tool(rag_factory))
    return tools
