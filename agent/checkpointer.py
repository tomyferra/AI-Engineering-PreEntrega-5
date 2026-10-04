from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import CHECKPOINTS_DB_PATH


@asynccontextmanager
async def open_checkpointer(db_path: Path = CHECKPOINTS_DB_PATH) -> AsyncIterator[AsyncSqliteSaver]:
    """Checkpointer SQLite persistente en disco.

    Se usa AsyncSqliteSaver (la variante asíncrona de SqliteSaver, del mismo paquete
    langgraph-checkpoint-sqlite) porque el grafo se ejecuta con ainvoke/astream: SqliteSaver
    solo implementa los métodos síncronos. Guarda una "foto" del estado después de cada nodo,
    indexada por thread_id: reabrir el archivo con el mismo thread_id recupera la conversación.
    """
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver
