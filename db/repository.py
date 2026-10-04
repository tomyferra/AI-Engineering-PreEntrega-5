"""Acceso asíncrono (aiosqlite) a la base de pedidos. Solo consultas parametrizadas de lectura:
el LLM nunca genera SQL, únicamente elige qué método llamar y con qué argumentos (Least Privilege)."""
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

import aiosqlite

from config import PEDIDOS_DB_PATH
from schemas import Cliente, ItemPedido, Pedido, Producto


class PedidosRepository:
    def __init__(self, db_path: Path = PEDIDOS_DB_PATH):
        if not db_path.exists():
            raise FileNotFoundError(f"No existe la base de pedidos {db_path}: corré 'python setup_db.py'")
        # mode=ro: la conexión es de solo lectura, aunque una tool quisiera escribir no podría.
        self._uri = f"file:{quote(db_path.resolve().as_posix(), safe='/:')}?mode=ro"

    async def _fetch(self, sql: str, params: tuple = ()) -> List[aiosqlite.Row]:
        async with aiosqlite.connect(self._uri, uri=True) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql, params) as cursor:
                return list(await cursor.fetchall())

    async def listar_productos(self) -> List[Producto]:
        filas = await self._fetch(
            "SELECT producto_id, nombre, unidades_por_pack, medida, precio_lista, stock_packs FROM productos"
        )
        return [Producto(**dict(f)) for f in filas]

    async def listar_clientes(self) -> List[Cliente]:
        filas = await self._fetch("SELECT * FROM clientes ORDER BY cliente_id")
        return [Cliente(**dict(f)) for f in filas]

    async def obtener_cliente(self, cliente_id: int) -> Optional[Cliente]:
        filas = await self._fetch("SELECT * FROM clientes WHERE cliente_id = ?", (cliente_id,))
        return Cliente(**dict(filas[0])) if filas else None

    async def buscar_clientes(self, nombre: str) -> List[Cliente]:
        filas = await self._fetch(
            "SELECT * FROM clientes WHERE nombre LIKE ? ORDER BY nombre", (f"%{nombre}%",)
        )
        return [Cliente(**dict(f)) for f in filas]

    async def pedidos_de_cliente(self, cliente_id: int, estado: Optional[str] = None,
                                 desde: Optional[str] = None, hasta: Optional[str] = None) -> List[Pedido]:
        """Pedidos del cliente ordenados del más reciente al más antiguo (sin ítems)."""
        sql = "SELECT * FROM pedidos WHERE cliente_id = ?"
        params: list = [cliente_id]
        for condicion, valor in (("estado = ?", estado), ("fecha >= ?", desde), ("fecha <= ?", hasta)):
            if valor:
                sql += f" AND {condicion}"
                params.append(valor)
        filas = await self._fetch(sql + " ORDER BY fecha DESC, pedido_id DESC", tuple(params))
        return [Pedido(**dict(f)) for f in filas]

    async def obtener_pedido(self, pedido_id: int) -> Optional[Pedido]:
        filas = await self._fetch("SELECT * FROM pedidos WHERE pedido_id = ?", (pedido_id,))
        if not filas:
            return None
        items = await self._fetch(
            "SELECT producto_id, cantidad, precio_unitario FROM pedido_items WHERE pedido_id = ?", (pedido_id,)
        )
        return Pedido(**dict(filas[0]), items=[ItemPedido(**dict(i)) for i in items])
