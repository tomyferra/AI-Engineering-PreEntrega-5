from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pathlib import Path

# 1. El Estado: la "memoria de trabajo" que fluye por el grafo
class SupportState(TypedDict):
    query: str
    attempts: int
    solved: bool
    escalated: bool
# 2. Nodo: busca en la base de conocimiento (simulado)
def search_knowledge(state: SupportState) -> dict:
    intento = state["attempts"] + 1
    encontrado = intento >= 2          # simulamos que recién resuelve en el 2º intento
    print(f"[search] intento {intento} -> {'resuelto' if encontrado else 'sin resultado'}")
    return {"attempts": intento, "solved": encontrado}
# 3. Nodo: escala a un humano
def human_support(state: SupportState) -> dict:
    print("[human] caso escalado a soporte humano")
    return {"escalated": True}
# 4. Arista condicional: decide el camino según el estado
def route(state: SupportState) -> str:
    if state["solved"]:
        return "end"
    if state["attempts"] >= 3:         # corta el ciclo para no quedar infinito
        return "escalate"
    return "retry"                     # vuelve a 'search' -> esto crea el CICLO
# 5. Construcción del grafo
workflow = StateGraph(SupportState)
workflow.add_node("search", search_knowledge)
workflow.add_node("human", human_support)
workflow.add_edge(START, "search")
workflow.add_conditional_edges(
    "search",
    route,
    {"end": END, "escalate": "human", "retry": "search"},
)
workflow.add_edge("human", END)
app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path=Path(__file__).parent / "graph.png")

# 6. Ejecución
final = app.invoke(
    {"query": "No puedo iniciar sesión", "attempts": 0, "solved": False, "escalated": False}
)
print(final)