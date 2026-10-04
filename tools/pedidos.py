"""Tools del sistema de pedidos de GenPet (base SQLite simulada).

Se construyen con un factory que recibe el repositorio: la infraestructura (conexión a la DB) queda
fuera del State del grafo y las tools se pueden testear con una base temporal.

Los errores esperables (cliente inexistente, pedido inexistente) no lanzan excepciones: se devuelven
como un dict con "error" y una "accion_sugerida", para que el LLM pueda reintentar o pedir aclaraciones.
"""
import difflib
from datetime import date
from typing import List, Optional

from langchain_core.tools import BaseTool, tool

from db.repository import PedidosRepository
from formatting import ars
from schemas import BuscarClienteInput, ConsultarPedidosInput, DetallePedidoInput

MAX_SUGERENCIAS = 3
UMBRAL_SIMILITUD = 0.6


def similitud(consulta: str, nombre: str) -> float:
    """Similitud (0-1) entre lo buscado y un nombre, comparando contra cada tramo de palabras consecutivas
    del mismo largo: 'San Roke' se compara con 'Veterinaria San', 'San Roque', etc. y no con el nombre entero."""
    consulta, palabras = consulta.lower(), nombre.lower().split()
    n = min(len(consulta.split()), len(palabras))
    tramos = [" ".join(palabras[i:i + n]) for i in range(len(palabras) - n + 1)]
    return max(difflib.SequenceMatcher(None, consulta, t).ratio() for t in [nombre.lower(), *tramos])


def build_pedidos_tools(repo: PedidosRepository) -> List[BaseTool]:

    @tool
    async def listar_productos() -> dict:
        """Devuelve el catálogo actual de GenPet: los dos packs de paños higiénicos para mascotas de 60x90 cm
        (pack x20 y pack x50) con su código, precio de lista vigente en pesos y stock disponible en depósito.
        Usala para preguntas sobre precios actuales, stock o qué productos vende la empresa."""
        productos = await repo.listar_productos()
        return {"productos": [{**p.model_dump(), "precio_lista_ars": ars(p.precio_lista)} for p in productos]}

    @tool(args_schema=BuscarClienteInput)
    async def buscar_cliente(nombre: str) -> dict:
        """Busca clientes de GenPet por nombre (coincidencia parcial, sin distinguir mayúsculas) y devuelve su
        cliente_id, tipo (distribuidor, pet shop, veterinaria o consumidor final) y ciudad.
        Usala SIEMPRE primero cuando el usuario mencione un cliente por su nombre: las demás tools de pedidos
        necesitan el cliente_id numérico. Si no hay coincidencias, devuelve nombres parecidos como sugerencia."""
        clientes = await repo.buscar_clientes(nombre)
        if clientes:
            return {"encontrados": [c.model_dump() for c in clientes]}

        clientes = await repo.listar_clientes()
        puntajes = sorted(((similitud(nombre, c.nombre), c) for c in clientes), key=lambda x: x[0], reverse=True)
        sugerencias = [c.model_dump() for score, c in puntajes[:MAX_SUGERENCIAS] if score >= UMBRAL_SIMILITUD]
        return {
            "error": f"No existe ningún cliente cuyo nombre contenga '{nombre}'.",
            "sugerencias": sugerencias,
            "accion_sugerida": "Si alguna sugerencia coincide claramente con lo que pidió el usuario, reintentá "
                               "con ese nombre; si no, pedile al usuario que aclare el nombre del cliente.",
        }

    @tool(args_schema=ConsultarPedidosInput)
    async def consultar_pedidos_cliente(cliente_id: int, estado: Optional[str] = None,
                                        desde: Optional[date] = None, hasta: Optional[date] = None) -> dict:
        """Devuelve los pedidos de un cliente (del más reciente al más antiguo) con la cantidad de pedidos,
        el total facturado y, para cada pedido: número, fecha, canal, estado y total en pesos.
        Permite filtrar por estado y por rango de fechas. Requiere el cliente_id numérico (usar buscar_cliente
        antes si solo se conoce el nombre). Para ver qué productos tuvo un pedido, usar detalle_pedido."""
        cliente = await repo.obtener_cliente(cliente_id)
        if cliente is None:
            return {"error": f"No existe un cliente con id {cliente_id}.",
                    "accion_sugerida": "Usá buscar_cliente con el nombre del cliente para obtener su id correcto."}

        pedidos = await repo.pedidos_de_cliente(
            cliente_id, estado=estado,
            desde=desde.isoformat() if desde else None,
            hasta=hasta.isoformat() if hasta else None,
        )
        facturables = [p for p in pedidos if p.estado != "cancelado"]
        total = sum(p.total for p in facturables)
        return {
            "cliente": cliente.model_dump(),
            "filtros": {"estado": estado, "desde": desde and desde.isoformat(), "hasta": hasta and hasta.isoformat()},
            "cantidad_pedidos": len(pedidos),
            "cantidad_cancelados": len(pedidos) - len(facturables),
            "total_facturado": total,
            "total_facturado_ars": ars(total),
            "nota": "El total facturado excluye los pedidos cancelados.",
            "pedidos": [{"pedido_id": p.pedido_id, "fecha": p.fecha, "canal": p.canal, "estado": p.estado,
                         "total_ars": ars(p.total)} for p in pedidos],
        }

    @tool(args_schema=DetallePedidoInput)
    async def detalle_pedido(pedido_id: int) -> dict:
        """Devuelve el detalle completo de un pedido por su número: cliente, fecha, canal, estado, los productos
        comprados (pack x20 / pack x50) con cantidad y precio unitario, subtotal, descuento aplicado y total.
        Usala cuando el usuario pregunte qué se compró en un pedido puntual (por ejemplo, 'el último pedido')."""
        pedido = await repo.obtener_pedido(pedido_id)
        if pedido is None:
            return {"error": f"No existe el pedido {pedido_id}.",
                    "accion_sugerida": "Usá consultar_pedidos_cliente para ver los números de pedido válidos."}

        cliente = await repo.obtener_cliente(pedido.cliente_id)
        nombres = {p.producto_id: p.nombre for p in await repo.listar_productos()}
        return {
            "pedido_id": pedido.pedido_id,
            "cliente": cliente.model_dump() if cliente else pedido.cliente_id,
            "fecha": pedido.fecha,
            "canal": pedido.canal,
            "estado": pedido.estado,
            "items": [{"producto_id": i.producto_id, "producto": nombres.get(i.producto_id, i.producto_id),
                       "cantidad_packs": i.cantidad, "precio_unitario_ars": ars(i.precio_unitario),
                       "importe_ars": ars(i.cantidad * i.precio_unitario)} for i in pedido.items],
            "subtotal_ars": ars(pedido.subtotal),
            "descuento_pct": pedido.descuento_pct,
            "total_ars": ars(pedido.total),
        }

    return [listar_productos, buscar_cliente, consultar_pedidos_cliente, detalle_pedido]
