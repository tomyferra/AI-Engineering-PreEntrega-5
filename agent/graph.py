"""Construcción del grafo ReAct:

    START -> agente --(tools_condition)--> tools -> agente  (ciclo)
                     \\-> END  (cuando el LLM ya no pide herramientas)
"""
from typing import Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.nodes import make_agent_node
from agent.state import AgentState

AGENT_NODE = "agente"
TOOLS_NODE = "tools"


def build_graph(model: BaseChatModel, tools: Sequence[BaseTool],
                checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    workflow = StateGraph(AgentState)

    # bind_tools publica el contrato (nombre, docstring, args_schema) de cada tool al LLM.
    workflow.add_node(AGENT_NODE, make_agent_node(model.bind_tools(tools)))
    # handle_tool_errors=True: una excepción inesperada dentro de una tool (o argumentos inválidos según
    # el args_schema de Pydantic) vuelve al LLM como ToolMessage de error en lugar de romper el grafo.
    workflow.add_node(TOOLS_NODE, ToolNode(tools, handle_tool_errors=True))

    workflow.add_edge(START, AGENT_NODE)
    # tools_condition devuelve "tools" si el último AIMessage tiene tool_calls, o END si no.
    workflow.add_conditional_edges(AGENT_NODE, tools_condition, {TOOLS_NODE: TOOLS_NODE, END: END})
    # El resultado de las tools (observación) siempre vuelve al agente: esto cierra el ciclo ReAct.
    workflow.add_edge(TOOLS_NODE, AGENT_NODE)

    return workflow.compile(checkpointer=checkpointer)
