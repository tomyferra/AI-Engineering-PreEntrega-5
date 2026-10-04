from typing import Awaitable, Callable

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.runnables import Runnable

from agent.prompts import build_system_prompt
from agent.state import AgentState
from config import MAX_HISTORY_TOKENS

AgentNode = Callable[[AgentState], Awaitable[dict]]


def make_agent_node(model_with_tools: Runnable[LanguageModelInput, BaseMessage],
                    max_history_tokens: int = MAX_HISTORY_TOKENS) -> AgentNode:
    """Nodo "agente" (Reason): el LLM ve el historial y decide si llama herramientas o responde.

    Única responsabilidad: invocar al modelo. La ejecución de tools vive en el ToolNode y la
    decisión de ruta en tools_condition.
    """

    async def agente(state: AgentState) -> dict:
        # Evita el "estado sucio": el checkpoint guarda todo el historial, pero al LLM solo le enviamos
        # los mensajes más recientes que entren en el presupuesto. start_on="human" garantiza que el
        # recorte no deje un ToolMessage huérfano (sin el AIMessage que lo pidió).
        historial = trim_messages(
            state["messages"],
            max_tokens=max_history_tokens,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on="human",
        ) or state["messages"][-1:]
        respuesta = await model_with_tools.ainvoke([SystemMessage(build_system_prompt()), *historial])
        # Devolvemos solo la actualización: add_messages la anexa al historial.
        return {"messages": [respuesta]}

    return agente
