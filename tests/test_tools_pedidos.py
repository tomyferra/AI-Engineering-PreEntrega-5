import json

import pytest

from config import SEED_DIR

PEDIDOS_SEED = json.loads((SEED_DIR / "pedidos.json").read_text(encoding="utf-8"))


def esperado_cliente(cliente_id: int) -> tuple[int, int]:
    pedidos = [p for p in PEDIDOS_SEED if p["cliente_id"] == cliente_id]
    total = sum(p["total"] for p in pedidos if p["estado"] != "cancelado")
    return len(pedidos), total


async def test_listar_productos_devuelve_los_dos_packs(tools_by_name):
    resultado = await tools_by_name["listar_productos"].ainvoke({})
    assert {p["producto_id"] for p in resultado["productos"]} == {"GP-20", "GP-50"}


async def test_buscar_cliente_por_nombre_parcial(tools_by_name):
    resultado = await tools_by_name["buscar_cliente"].ainvoke({"nombre": "patitas"})
    assert [c["cliente_id"] for c in resultado["encontrados"]] == [101]


async def test_buscar_cliente_mal_escrito_devuelve_error_con_sugerencias(tools_by_name):
    resultado = await tools_by_name["buscar_cliente"].ainvoke({"nombre": "veterinaria San Roke"})
    assert "error" in resultado
    nombres = [s["nombre"] for s in resultado["sugerencias"]]
    assert nombres[0] == "Veterinaria San Roque", "la sugerencia más parecida va primero"
    assert "Martín Ruiz" not in nombres
    assert "accion_sugerida" in resultado


async def test_buscar_cliente_sin_parecidos_no_sugiere_cualquier_cosa(tools_by_name):
    resultado = await tools_by_name["buscar_cliente"].ainvoke({"nombre": "Zapatería XYZ"})
    assert resultado["sugerencias"] == []


async def test_consultar_pedidos_coincide_con_el_seed(tools_by_name):
    resultado = await tools_by_name["consultar_pedidos_cliente"].ainvoke({"cliente_id": 101})
    cantidad, total = esperado_cliente(101)
    assert resultado["cantidad_pedidos"] == cantidad
    assert resultado["total_facturado"] == total
    fechas = [p["fecha"] for p in resultado["pedidos"]]
    assert fechas == sorted(fechas, reverse=True), "el primer pedido debe ser el más reciente"


async def test_consultar_pedidos_filtra_por_estado_y_fecha(tools_by_name):
    resultado = await tools_by_name["consultar_pedidos_cliente"].ainvoke(
        {"cliente_id": 103, "estado": "entregado", "desde": "2026-07-01", "hasta": "2026-09-30"}
    )
    assert all(p["estado"] == "entregado" and "2026-07-01" <= p["fecha"] <= "2026-09-30"
               for p in resultado["pedidos"])


async def test_consultar_pedidos_de_cliente_inexistente_devuelve_error(tools_by_name):
    resultado = await tools_by_name["consultar_pedidos_cliente"].ainvoke({"cliente_id": 999})
    assert "error" in resultado


async def test_args_invalidos_son_rechazados_por_pydantic(tools_by_name):
    with pytest.raises(Exception):
        await tools_by_name["consultar_pedidos_cliente"].ainvoke({"cliente_id": "no-es-un-numero"})


async def test_detalle_pedido_incluye_items(tools_by_name):
    pedido = PEDIDOS_SEED[0]
    resultado = await tools_by_name["detalle_pedido"].ainvoke({"pedido_id": pedido["pedido_id"]})
    assert len(resultado["items"]) == len(pedido["items"])
    assert resultado["descuento_pct"] == pedido["descuento_pct"]


async def test_repositorio_es_de_solo_lectura(repo):
    import aiosqlite
    async with aiosqlite.connect(repo._uri, uri=True) as conn:
        with pytest.raises(aiosqlite.OperationalError):
            await conn.execute("DELETE FROM pedidos")
