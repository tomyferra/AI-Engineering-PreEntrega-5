"""Punto único de configuración: rutas, parámetros del agente/RAG y settings leídos desde .env."""
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, SecretStr

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
SEED_DIR = DATA_DIR / "seed"
CATALOG_PATH = DATA_DIR / "catalog.json"
TRACES_DIR = BASE_DIR / "traces"

# Base de datos de negocio (clientes/pedidos) y base de checkpoints del agente: archivos separados,
# así borrar la memoria del agente nunca toca los datos de la empresa (y viceversa).
PEDIDOS_DB_PATH = DATA_DIR / "genpet.db"
CHECKPOINTS_DB_PATH = BASE_DIR / "checkpoints.sqlite"

# --- Agente ---
# Techo de pasos del grafo por invocación: evita bucles infinitos y costos inesperados en la API.
RECURSION_LIMIT = 30
# Máximo de tokens del historial que se envían al LLM. El estado conserva todo; solo se recorta
# lo que viaja en cada llamada (evita el "estado sucio" que desborda la ventana de contexto).
# Los resultados de las tools (listas de pedidos, fragmentos del RAG) son largos: un presupuesto chico
# hace que el agente "olvide" los primeros turnos aunque sigan guardados en el checkpoint.
MAX_HISTORY_TOKENS = 16000

# --- RAG ---
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60
TOP_K = 4
# Cada retriever trae más candidatos que TOP_K para que la fusión (RRF) tenga de dónde elegir.
CANDIDATES_K = 8
# Pesos del EnsembleRetriever: [BM25 (léxico), Pinecone (semántico)].
ENSEMBLE_WEIGHTS = [0.5, 0.5]


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


DEFAULT_MODEL_POR_PROVIDER = {
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-haiku-4-5",
    LLMProvider.GEMINI: "gemini-2.5-flash",
    LLMProvider.OPENROUTER: "openai/gpt-4o-mini",
}

ENV_VAR_POR_LLM_PROVIDER = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.GEMINI: "GEMINI_API_KEY",
    LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
}

EMBEDDING_MODELS_POR_PROVIDER = {
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.GEMINI: "models/gemini-embedding-001",
}

ENV_VAR_POR_EMBEDDING_PROVIDER = {
    EmbeddingProvider.OPENAI: "OPENAI_API_KEY",
    EmbeddingProvider.GEMINI: "GEMINI_API_KEY",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _require_env(nombre: str) -> str:
    valor = os.getenv(nombre)
    if not valor:
        raise ValueError(f"Falta la variable de entorno {nombre} (ver .env.example)")
    return valor


class LLMSettings(BaseModel):
    provider: LLMProvider
    model: str
    api_key: SecretStr
    temperature: float = Field(default=0.0, ge=0, le=2)


class RAGSettings(BaseModel):
    pinecone_api_key: SecretStr
    index_name: str
    namespace: str
    cloud: str = "aws"
    region: str = "us-east-1"
    metric: str = "cosine"
    # text-embedding-3-small es nativo de 1536; gemini-embedding-001 se trunca a 1536 (MRL).
    dimension: int = Field(default=1536, gt=0)
    embedding_provider: EmbeddingProvider = EmbeddingProvider.GEMINI

    @property
    def embedding_model(self) -> str:
        return EMBEDDING_MODELS_POR_PROVIDER[self.embedding_provider]

    def get_embedding_api_key(self) -> str:
        return _require_env(ENV_VAR_POR_EMBEDDING_PROVIDER[self.embedding_provider])


def get_llm_settings(provider: Optional[str] = None) -> LLMSettings:
    """Settings del LLM del agente (llamar después de load_dotenv). LLM_MODEL pisa el modelo por defecto."""
    proveedor = LLMProvider((provider or os.getenv("LLM_PROVIDER", "openai")).lower())
    return LLMSettings(
        provider=proveedor,
        model=os.getenv("LLM_MODEL") or DEFAULT_MODEL_POR_PROVIDER[proveedor],
        api_key=SecretStr(_require_env(ENV_VAR_POR_LLM_PROVIDER[proveedor])),
    )


def get_rag_settings() -> RAGSettings:
    """Settings de Pinecone + embeddings (llamar después de load_dotenv)."""
    return RAGSettings(
        pinecone_api_key=SecretStr(_require_env("PINECONE_API_KEY")),
        index_name=_require_env("INDEX_NAME"),
        namespace=os.getenv("PINECONE_NAMESPACE", "genpet-docs"),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
        embedding_provider=EmbeddingProvider(os.getenv("EMBEDDING_PROVIDER", "gemini").lower()),
    )
