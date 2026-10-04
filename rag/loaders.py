# Adaptado de la Pre-entrega 4 (rag/loaders.py): el corpus de GenPet es solo Markdown.
import json
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from config import CATALOG_PATH, DOCS_DIR
from schemas import DocumentMeta

CATEGORIA_POR_DEFECTO = "general"


def load_catalog(path: Path = CATALOG_PATH) -> Dict[str, dict]:
    """Catálogo {doc_id: {categoria, tags}} con las etiquetas de cada documento."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_documents(docs_dir: Path = DOCS_DIR, catalog_path: Path = CATALOG_PATH) -> List[Document]:
    """Carga los Markdown de docs_dir con su metadata (fuente, categoría y tags del catálogo)."""
    catalog = load_catalog(catalog_path)
    documentos: List[Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        entrada = catalog.get(path.stem, {})
        meta = DocumentMeta(
            doc_id=path.stem,
            source=path.name,
            categoria=entrada.get("categoria", CATEGORIA_POR_DEFECTO),
            tags=entrada.get("tags", []),
        )
        documentos.append(Document(page_content=path.read_text(encoding="utf-8"), metadata=meta.model_dump()))
    if not documentos:
        raise FileNotFoundError(f"No se encontraron archivos .md en {docs_dir}")
    return documentos
