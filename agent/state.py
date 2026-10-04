from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Estado del agente: hereda de MessagesState.

    MessagesState ya define `messages: Annotated[list[AnyMessage], add_messages]`: el reducer
    add_messages anexa los mensajes nuevos en lugar de sobrescribir la lista, así el historial
    (pregunta -> tool calls -> observaciones -> respuesta) crece con cada paso del ciclo ReAct.

    No se agregan más campos a propósito: el estado solo guarda datos que deben persistir en el
    checkpoint. La DB, el LLM y el retriever se inyectan al construir el grafo (no viven en el State).
    """
