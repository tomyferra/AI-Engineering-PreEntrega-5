"""Pipeline de ingesta del RAG de GenPet: data/docs -> chunks -> embeddings -> Pinecone. Uso: python ingest.py [--force]"""
import logging
import sys

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

from config import get_rag_settings
from rag.ingestion import ingest

if __name__ == "__main__":
    ingest(get_rag_settings(), force="--force" in sys.argv)
