"""Connexion SQLite WAL, FK activées, connexion thread-locale.

Chemin de la BD :
  - En développement  : répertoire racine du projet (detect_pick/)
  - En production .exe : répertoire du .exe (PyInstaller frozen)
  Dans les deux cas, beac.db et cache/ survivent aux mises à jour.
"""
import sys
import os
import sqlite3
from contextlib import contextmanager
from threading import local


def _base_dir() -> str:
    """Répertoire persistant : à côté du .exe en prod, à la racine du projet en dev."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_BASE       = _base_dir()
DB_PATH     = os.path.join(_BASE, "beac.db")
FIGURES_DIR = os.path.join(_BASE, "cache", "figures")

_local = local()


def get_connection() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


@contextmanager
def db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def close_connection() -> None:
    """Ferme la connexion SQLite du thread courant, si elle existe.

    À appeler en fin de thread de calcul de courte durée (pipeline,
    optimisation) pour ne pas laisser traîner des handles WAL ouverts au-delà
    de la vie utile du thread.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        finally:
            _local.conn = None
