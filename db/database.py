"""Esquema y carga inicial de la base SQLite de pedidos de GenPet (data/seed/*.json -> data/genpet.db)."""
import json
import logging
import sqlite3
from pathlib import Path

from config import PEDIDOS_DB_PATH, SEED_DIR

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS productos (
    producto_id       TEXT PRIMARY KEY,
    nombre            TEXT NOT NULL,
    unidades_por_pack INTEGER NOT NULL,
    medida            TEXT NOT NULL,
    precio_lista      INTEGER NOT NULL,
    costo_unitario    INTEGER NOT NULL,
    stock_packs       INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS clientes (
    cliente_id INTEGER PRIMARY KEY,
    nombre     TEXT NOT NULL,
    tipo       TEXT NOT NULL,
    ciudad     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pedidos (
    pedido_id     INTEGER PRIMARY KEY,
    cliente_id    INTEGER NOT NULL REFERENCES clientes(cliente_id),
    fecha         TEXT NOT NULL,
    canal         TEXT NOT NULL,
    estado        TEXT NOT NULL,
    descuento_pct INTEGER NOT NULL,
    subtotal      INTEGER NOT NULL,
    total         INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pedido_items (
    pedido_id       INTEGER NOT NULL REFERENCES pedidos(pedido_id),
    producto_id     TEXT NOT NULL REFERENCES productos(producto_id),
    cantidad        INTEGER NOT NULL,
    precio_unitario INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pedidos_cliente ON pedidos(cliente_id, fecha);
"""


def _load_seed(nombre: str, seed_dir: Path) -> list:
    path = seed_dir / f"{nombre}.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}: corré primero 'python generate_mock_data.py'")
    return json.loads(path.read_text(encoding="utf-8"))


def create_database(db_path: Path = PEDIDOS_DB_PATH, seed_dir: Path = SEED_DIR, force: bool = False) -> bool:
    """Crea la base y la puebla con los JSON de seed. Devuelve False si ya existía y no se pidió force."""
    if db_path.exists():
        if not force:
            logger.info("La base '%s' ya existe; se omite la carga (usá --force para recrearla).", db_path)
            return False
        db_path.unlink()

    productos = _load_seed("productos", seed_dir)
    clientes = _load_seed("clientes", seed_dir)
    pedidos = _load_seed("pedidos", seed_dir)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        # El precio vigente hoy es el de la lista actualizada en julio.
        conn.executemany(
            "INSERT INTO productos VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(p["producto_id"], p["nombre"], p["unidades_por_pack"], p["medida"],
              p["precio_lista_desde_julio"], p["costo_unitario"], p["stock_packs"]) for p in productos],
        )
        conn.executemany(
            "INSERT INTO clientes VALUES (:cliente_id, :nombre, :tipo, :ciudad)", clientes
        )
        conn.executemany(
            "INSERT INTO pedidos VALUES (:pedido_id, :cliente_id, :fecha, :canal, :estado, "
            ":descuento_pct, :subtotal, :total)",
            pedidos,
        )
        conn.executemany(
            "INSERT INTO pedido_items VALUES (?, ?, ?, ?)",
            [(p["pedido_id"], i["producto_id"], i["cantidad"], i["precio_unitario"])
             for p in pedidos for i in p["items"]],
        )
    logger.info("Base '%s' creada: %d productos, %d clientes, %d pedidos.",
                db_path, len(productos), len(clientes), len(pedidos))
    return True
