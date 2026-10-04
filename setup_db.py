"""Crea la base SQLite de pedidos a partir de data/seed. Uso: python setup_db.py [--force]"""
import logging
import sys

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from db.database import create_database

if __name__ == "__main__":
    create_database(force="--force" in sys.argv)
