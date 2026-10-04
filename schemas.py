"""Modelos Pydantic: entidades del negocio, contratos de entrada de las tools y metadata del RAG."""
from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

EstadoPedido = Literal["en preparación", "despachado", "entregado", "cancelado"]


# --- Entidades del negocio (lo que devuelve el repositorio de pedidos) ---

class Cliente(BaseModel):
    cliente_id: int
    nombre: str
    tipo: str
    ciudad: str


class Producto(BaseModel):
    producto_id: str
    nombre: str
    unidades_por_pack: int
    medida: str
    precio_lista: int = Field(description="Precio de lista vigente en ARS")
    stock_packs: int


class ItemPedido(BaseModel):
    producto_id: str
    cantidad: int = Field(gt=0)
    precio_unitario: int


class Pedido(BaseModel):
    pedido_id: int
    cliente_id: int
    fecha: str
    canal: str
    estado: EstadoPedido
    descuento_pct: int
    subtotal: int
    total: int
    items: List[ItemPedido] = Field(default_factory=list)


# --- Contratos de entrada de las tools: Pydantic valida lo que manda el LLM antes de tocar la DB ---

class BuscarClienteInput(BaseModel):
    nombre: str = Field(description="Nombre (o parte del nombre) del cliente, por ejemplo 'Patitas' o 'San Roque'")

    @field_validator("nombre")
    @classmethod
    def nombre_no_vacio(cls, v: str) -> str:
        if len(v.strip()) < 2:
            raise ValueError("el nombre debe tener al menos 2 caracteres")
        return v.strip()


class ConsultarPedidosInput(BaseModel):
    cliente_id: int = Field(ge=1, description="ID numérico del cliente (obtenerlo antes con buscar_cliente)")
    estado: Optional[EstadoPedido] = Field(
        default=None, description="Filtrar por estado del pedido; omitir para traer todos"
    )
    desde: Optional[date] = Field(default=None, description="Fecha inicial inclusive, formato AAAA-MM-DD")
    hasta: Optional[date] = Field(default=None, description="Fecha final inclusive, formato AAAA-MM-DD")


class DetallePedidoInput(BaseModel):
    pedido_id: int = Field(ge=1, description="Número de pedido, por ejemplo 5123")


class BuscarInformacionInput(BaseModel):
    consulta: str = Field(
        description="Pregunta concreta en lenguaje natural sobre la empresa, por ejemplo "
                    "'ganancia neta del Q2 2026' o 'política de devoluciones'"
    )


# --- RAG ---

class DocumentMeta(BaseModel):
    """Metadatos que viajan con cada chunk a Pinecone (el texto va aparte, en la clave 'text')."""

    doc_id: str
    source: str
    categoria: str
    tags: List[str] = Field(default_factory=list)
    chunk_index: int = 0
