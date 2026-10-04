from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Estado: el borrador, su puntaje de calidad y cuántas veces lo revisamos
class WriteState(TypedDict):
    tema: str
    borrador: str
    score: int
    revisiones: int

# 2. Nodo escritor: genera/mejora el borrador (simulado: suma 3 puntos por vuelta)
def escribir(state: WriteState) -> dict:
    nuevo_score = state["score"] + 3
    print(f"[escritor] revisión {state['revisiones'] + 1} -> score {nuevo_score}")
    return {
        "borrador": f"Borrador sobre '{state['tema']}' (calidad {nuevo_score}/10)",
        "score": nuevo_score,
        "revisiones": state["revisiones"] + 1,
    }

# 3. Nodo crítico: no modifica el estado; la decisión la toma la arista condicional
def criticar(state: WriteState) -> dict:
    return {}

# 4. Arista condicional: ¿aprobamos, nos rendimos o seguimos revisando?
def route(state: WriteState) -> str:
    if state["score"] >= 8:
        return "aprobado"
    if state["revisiones"] >= 4:        # tope para no iterar infinito
        return "rendirse"
    return "revisar"                    # vuelve al escritor -> CICLO de reflexión

# 5. Construcción del grafo
g = StateGraph(WriteState)
g.add_node("escritor", escribir)
g.add_node("critico", criticar)

g.add_edge(START, "escritor")
g.add_edge("escritor", "critico")
g.add_conditional_edges(
    "critico",
    route,
    {"aprobado": END, "rendirse": END, "revisar": "escritor"},
)

app = g.compile()

# 6. Ejecución
final = app.invoke({"tema": "onboarding de usuarios", "borrador": "", "score": 0, "revisiones": 0})
print(f"Final tras {final['revisiones']} revisiones | score {final['score']}")
print(final["borrador"])
