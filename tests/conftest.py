from typing import Any, List

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import generate_mock_data
from config import SEED_DIR
from db.database import create_database
from db.repository import PedidosRepository
from tools.pedidos import build_pedidos_tools


class FakeToolCallingModel(FakeMessagesListChatModel):
    """LLM falso con respuestas guionadas: permite testear el grafo sin API keys ni costos."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        # Copia por llamada: si se devolviera siempre el mismo objeto, add_messages lo deduplicaría por id.
        resultado = super()._generate(*args, **kwargs)
        mensaje = resultado.generations[0].message.model_copy(deep=True)
        mensaje.id = None
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


def tool_call(nombre: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": args, "id": call_id}])


def pytest_configure(config) -> None:
    # data/seed/ no se versiona: si falta (repo recién clonado), se genera con la semilla fija.
    if not (SEED_DIR / "pedidos.json").exists():
        generate_mock_data.main()


@pytest.fixture
def repo(tmp_path) -> PedidosRepository:
    db_path = tmp_path / "genpet_test.db"
    create_database(db_path=db_path)
    return PedidosRepository(db_path)


@pytest.fixture
def tools(repo) -> List:
    return build_pedidos_tools(repo)


@pytest.fixture
def tools_by_name(tools) -> dict:
    return {t.name: t for t in tools}
