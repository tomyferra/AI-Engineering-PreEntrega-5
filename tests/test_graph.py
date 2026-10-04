import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.checkpointer import open_checkpointer
from agent.graph import build_graph
from agent.session import run_config, run_turn
from agent.trace import TraceRecorder
from tests.conftest import FakeToolCallingModel, tool_call


async def test_ciclo_react_multipaso(tools):
    """El agente llama dos tools encadenadas y recién después responde (agente -> tools -> agente x2 -> END)."""
    model = FakeToolCallingModel(responses=[
        tool_call("buscar_cliente", {"nombre": "Patitas"}, "c1"),
        tool_call("consultar_pedidos_cliente", {"cliente_id": 101}, "c2"),
        AIMessage(content="Patitas Felices tuvo N pedidos."),
    ])
    graph = build_graph(model, tools)
    recorder = TraceRecorder(echo=None)

    respuesta = await run_turn(graph, "t", "¿Cuántos pedidos tuvo Patitas?", recorder)

    assert respuesta == "Patitas Felices tuvo N pedidos."
    turno = recorder.turnos[0]
    assert turno["llamadas_a_tools"] == 2
    assert [p["tipo"] for p in turno["pasos"]] == ["accion", "observacion", "accion", "observacion",
                                                  "respuesta_final"]


async def test_error_de_tool_vuelve_al_agente_como_observacion(tools):
    """Argumentos inválidos no rompen el grafo: el ToolNode devuelve el error y el agente puede reintentar."""
    model = FakeToolCallingModel(responses=[
        tool_call("consultar_pedidos_cliente", {"cliente_id": "abc"}, "c1"),
        tool_call("consultar_pedidos_cliente", {"cliente_id": 101}, "c2"),
        AIMessage(content="Listo."),
    ])
    graph = build_graph(model, tools)

    resultado = await graph.ainvoke({"messages": [HumanMessage("pedidos del cliente abc")]})

    tool_messages = [m for m in resultado["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].status == "error"
    assert tool_messages[1].status == "success"
    assert resultado["messages"][-1].content == "Listo."


async def test_memoria_persistente_por_thread_id(tools, tmp_path):
    """El historial sobrevive a cerrar y reabrir el checkpointer SQLite, y queda aislado por thread_id."""
    db = tmp_path / "checkpoints.sqlite"

    async with open_checkpointer(db) as checkpointer:
        graph = build_graph(FakeToolCallingModel(responses=[AIMessage(content="Hola Ana")]), tools, checkpointer)
        await graph.ainvoke({"messages": [HumanMessage("Soy Ana")]}, config=run_config("thread-a"))

    # "Reinicio": nuevo checkpointer, nuevo grafo, mismo archivo.
    async with open_checkpointer(db) as checkpointer:
        graph = build_graph(FakeToolCallingModel(responses=[AIMessage(content="Te llamás Ana")]), tools, checkpointer)
        resultado = await graph.ainvoke({"messages": [HumanMessage("¿Cómo me llamo?")]},
                                        config=run_config("thread-a"))
        assert [m.content for m in resultado["messages"]] == ["Soy Ana", "Hola Ana", "¿Cómo me llamo?",
                                                              "Te llamás Ana"]

        otro = await graph.aget_state(run_config("thread-b"))
        assert otro.values == {}


async def test_traza_se_acumula_entre_ejecuciones(tools, tmp_path):
    """Como hace main.py: cada ejecución carga la traza previa del thread y le agrega sus turnos."""
    json_path, log_path = tmp_path / "t.json", tmp_path / "t.log"
    for respuesta in ("uno", "dos"):
        graph = build_graph(FakeToolCallingModel(responses=[AIMessage(content=respuesta)]), tools)
        recorder = TraceRecorder(echo=None, json_path=json_path, log_path=log_path)
        await run_turn(graph, "t", "hola", recorder)

    turnos = json.loads(json_path.read_text(encoding="utf-8"))["turnos"]
    assert [(t["turno"], t["respuesta"]) for t in turnos] == [(1, "uno"), (2, "dos")]
    assert log_path.read_text(encoding="utf-8").count("[turno ") == 2


async def test_recursion_limit_corta_un_bucle_infinito(tools):
    """Un LLM que nunca deja de pedir tools es frenado por recursion_limit en vez de gastar tokens sin fin."""
    model = FakeToolCallingModel(responses=[tool_call("listar_productos", {}, "c1")])
    graph = build_graph(model, tools)
    recorder = TraceRecorder(echo=None)

    respuesta = await run_turn(graph, "t", "loop", recorder, recursion_limit=5)

    assert "límite de 5 pasos" in respuesta
    assert recorder.turnos[0]["error"]
