"""Initialisation BD idempotent, appelé une seule fois au démarrage."""
import os
import sqlite3
import hashlib
import logging
from .db import get_connection, FIGURES_DIR

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'analyste',
    email         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_login    TEXT,
    is_active     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sessions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token     TEXT NOT NULL UNIQUE,
    login_at  TEXT NOT NULL DEFAULT (datetime('now')),
    logout_at TEXT
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),
    pays        TEXT NOT NULL,
    volet       TEXT NOT NULL,
    modele      TEXT NOT NULL,
    params_json TEXT,
    duration_ms INTEGER,
    n_anomalies INTEGER,
    tau_mean    REAL,
    status      TEXT NOT NULL DEFAULT 'running',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS anomaly_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    period       TEXT,
    lof_score    REAL,
    bivat_score  REAL,
    hybrid_score REAL,
    is_anomaly   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS figure_cache (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    fig_name   TEXT NOT NULL,
    file_path  TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_id, fig_name)
);

CREATE TABLE IF NOT EXISTS result_cache (
    cache_key  TEXT PRIMARY KEY,
    run_id     INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER REFERENCES analysis_runs(id),
    user_id     INTEGER REFERENCES users(id),
    report_type TEXT,
    file_path   TEXT,
    params_json TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_prefs (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key        TEXT    NOT NULL,
    value      TEXT,
    updated_at TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time   TEXT    NOT NULL DEFAULT (datetime('now')),
    user_id      INTEGER REFERENCES users(id),
    action       TEXT    NOT NULL,
    table_name   TEXT,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS optimization_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER REFERENCES users(id),
    model        TEXT NOT NULL,
    mode         TEXT NOT NULL DEFAULT 'genetic',
    pays         TEXT,
    volet        TEXT,
    best_params  TEXT,
    best_score   REAL,
    n_generations INTEGER,
    duration_ms  INTEGER,
    status       TEXT NOT NULL DEFAULT 'running',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS anomaly_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    anomaly_id INTEGER NOT NULL REFERENCES anomaly_results(id) ON DELETE CASCADE,
    user_id    INTEGER REFERENCES users(id),
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS notifications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES users(id),
    run_id     INTEGER REFERENCES analysis_runs(id),
    type       TEXT NOT NULL,
    title      TEXT NOT NULL,
    body       TEXT,
    is_read    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_audit_log_event_time
    ON audit_log(event_time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_event
    ON audit_log(user_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_created_at
    ON analysis_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_results_run
    ON anomaly_results(run_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_comments_anomaly
    ON anomaly_comments(anomaly_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user_read
    ON notifications(user_id, is_read);
"""

# Colonnes ajoutées après coup à des tables existantes — SQLite n'a pas de
# "ADD COLUMN IF NOT EXISTS", chaque ALTER est donc tenté puis ignoré s'il
# existe déjà (idempotent, ne touche jamais aux données déjà présentes).
_COLUMN_MIGRATIONS = [
    ("anomaly_results", "status",           "TEXT DEFAULT 'new'"),
    ("anomaly_results", "assignee_user_id", "INTEGER"),
    ("anomaly_results", "reviewed_by",      "INTEGER"),
    ("anomaly_results", "reviewed_at",      "TEXT"),
]


def _migrate_columns(conn) -> None:
    for table, column, decl in _COLUMN_MIGRATIONS:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        except sqlite3.OperationalError:
            pass  # colonne déjà présente

_INITIAL_USERS = [
    ("analyste.beac", hashlib.sha256(b"beac2024").hexdigest(),     "analyste"),
    ("admin.beac",    hashlib.sha256(b"beacadmin2024").hexdigest(), "admin"),
]


def init_db() -> None:
    """Crée les tables et insère les utilisateurs par défaut si absents."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    conn = get_connection()
    conn.executescript(_DDL)
    _migrate_columns(conn)
    for username, pwd_hash, role in _INITIAL_USERS:
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?,?,?)",
            (username, pwd_hash, role),
        )
    conn.commit()
    log.info("BD initialisée : beac.db")
