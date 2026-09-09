"""Fonctions CRUD couche d'accès aux données."""
import json
import uuid
import os
import hashlib
import logging
from datetime import datetime, timedelta
from .db import db, FIGURES_DIR

log = logging.getLogger(__name__)


# ── Utilisateurs / Auth ───────────────────────────────────────────────────────

def get_user(username: str) -> dict | None:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1", (username,)
        ).fetchone()
        return dict(row) if row else None


def change_password(username: str, new_hash: str) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE username=? AND is_active=1",
            (new_hash, username),
        )


def create_session(user_id: int) -> str:
    token = str(uuid.uuid4())
    with db() as conn:
        conn.execute(
            "INSERT INTO sessions (user_id, token) VALUES (?,?)", (user_id, token)
        )
        conn.execute(
            "UPDATE users SET last_login=datetime('now') WHERE id=?", (user_id,)
        )
    return token


def close_session(token: str) -> None:
    if not token:
        return
    with db() as conn:
        conn.execute(
            "UPDATE sessions SET logout_at=datetime('now') WHERE token=?", (token,)
        )


def get_session_info(token: str) -> dict | None:
    if not token:
        return None
    with db() as conn:
        row = conn.execute(
            """SELECT s.login_at, u.username, u.role, u.id AS user_id
               FROM sessions s JOIN users u ON s.user_id=u.id
               WHERE s.token=? AND s.logout_at IS NULL""",
            (token,),
        ).fetchone()
        return dict(row) if row else None


# ── Runs d'analyse ────────────────────────────────────────────────────────────

def create_run(user_id: int | None, pays: str, volet: str,
               modele: str, params: dict) -> int:
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO analysis_runs (user_id, pays, volet, modele, params_json)
               VALUES (?,?,?,?,?)""",
            (user_id, pays, volet, modele, json.dumps(params)),
        )
        return cur.lastrowid


def finish_run(run_id: int, status: str, duration_ms: int,
               n_anomalies: int | None, tau_mean: float | None) -> None:
    with db() as conn:
        conn.execute(
            """UPDATE analysis_runs
               SET status=?, duration_ms=?, n_anomalies=?, tau_mean=?
               WHERE id=?""",
            (status, duration_ms, n_anomalies, tau_mean, run_id),
        )


def save_anomaly_rows(run_id: int, rows: list[dict]) -> None:
    if not rows:
        return
    with db() as conn:
        conn.executemany(
            """INSERT INTO anomaly_results
               (run_id, period, lof_score, bivat_score, hybrid_score, is_anomaly)
               VALUES (:run_id,:period,:lof_score,:bivat_score,:hybrid_score,:is_anomaly)""",
            [{"run_id": run_id, "period": r.get("period"),
              "lof_score": r.get("lof_score"), "bivat_score": r.get("bivat_score"),
              "hybrid_score": r.get("hybrid_score"),
              "is_anomaly": int(bool(r.get("is_anomaly", False)))}
             for r in rows],
        )


def get_run_history(limit: int = 50) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            """SELECT r.id, r.pays, r.volet, r.modele, r.status,
                      r.duration_ms, r.n_anomalies, r.tau_mean, r.created_at,
                      u.username
               FROM analysis_runs r LEFT JOIN users u ON r.user_id=u.id
               ORDER BY r.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_run_by_id(run_id: int) -> dict | None:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_runs WHERE id=?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


def get_latest_run(pays: str | None = None, modele: str | None = None) -> dict | None:
    """Dernier run 'success', filtré par pays/modèle si fournis (pour la génération de rapport)."""
    query = "SELECT * FROM analysis_runs WHERE status='success'"
    params: list = []
    if pays and pays != "all":
        query += " AND pays=?"
        params.append(pays)
    if modele and modele != "both":
        query += " AND modele=?"
        params.append(modele)
    query += " ORDER BY created_at DESC LIMIT 1"
    with db() as conn:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None


def get_run_anomalies(run_id: int) -> list[dict]:
    """Anomalies d'un run + nombre de commentaires par ligne."""
    with db() as conn:
        rows = conn.execute(
            """SELECT a.*, ru.username AS reviewer,
                      au.username AS assignee,
                      (SELECT COUNT(*) FROM anomaly_comments c WHERE c.anomaly_id = a.id) AS n_comments
               FROM anomaly_results a
               LEFT JOIN users ru ON a.reviewed_by = ru.id
               LEFT JOIN users au ON a.assignee_user_id = au.id
               WHERE a.run_id=?
               ORDER BY a.is_anomaly DESC, a.period""",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def _get_anomalous_periods(run_id: int) -> set:
    """Périodes en anomalie pour un run — sans le COUNT(*) commentaires de
    `get_run_anomalies` (inutile ici, seul le diff de périodes est exploité)."""
    with db() as conn:
        rows = conn.execute(
            "SELECT period FROM anomaly_results WHERE run_id=? AND is_anomaly=1",
            (run_id,),
        ).fetchall()
        return {r["period"] for r in rows}


def compare_runs(run_id_a: int, run_id_b: int) -> dict:
    """Métriques côte à côte + diff des périodes anomales entre deux runs."""
    run_a = get_run_by_id(run_id_a)
    run_b = get_run_by_id(run_id_b)
    anoms_a = _get_anomalous_periods(run_id_a)
    anoms_b = _get_anomalous_periods(run_id_b)
    return {
        "run_a": run_a, "run_b": run_b,
        "only_a": sorted(anoms_a - anoms_b),
        "only_b": sorted(anoms_b - anoms_a),
        "common": sorted(anoms_a & anoms_b),
    }


def update_anomaly_status(anomaly_id: int, status: str, reviewed_by: int | None) -> None:
    with db() as conn:
        conn.execute(
            """UPDATE anomaly_results
               SET status=?, reviewed_by=?, reviewed_at=datetime('now')
               WHERE id=?""",
            (status, reviewed_by, anomaly_id),
        )


def assign_anomaly(anomaly_id: int, assignee_user_id: int | None) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE anomaly_results SET assignee_user_id=? WHERE id=?",
            (assignee_user_id, anomaly_id),
        )


def add_anomaly_comment(anomaly_id: int, user_id: int | None, body: str) -> int:
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO anomaly_comments (anomaly_id, user_id, body) VALUES (?,?,?)",
            (anomaly_id, user_id, body),
        )
        return cur.lastrowid


def get_anomaly_comments(anomaly_id: int) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            """SELECT c.*, u.username FROM anomaly_comments c
               LEFT JOIN users u ON c.user_id = u.id
               WHERE c.anomaly_id=? ORDER BY c.created_at""",
            (anomaly_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Cache figures (sur disque) ────────────────────────────────────────────────

def save_figure(run_id: int, fig_name: str, fig_json: str) -> str:
    """Écrit le JSON Plotly sur disque, enregistre le chemin en BD."""
    fname = f"{run_id}_{fig_name}.json"
    path  = os.path.join(FIGURES_DIR, fname)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(fig_json)
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO figure_cache (run_id, fig_name, file_path) VALUES (?,?,?)",
            (run_id, fig_name, path),
        )
    return path


def load_figures_for_run(run_id: int) -> dict[str, str]:
    """Recharge les figures depuis disque utilisé depuis la page Historique."""
    with db() as conn:
        rows = conn.execute(
            "SELECT fig_name, file_path FROM figure_cache WHERE run_id=?", (run_id,)
        ).fetchall()
    figs: dict[str, str] = {}
    for row in rows:
        path = row["file_path"]
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                figs[row["fig_name"]] = fh.read()
    return figs


# ── Cache résultats (hash dataset+params) ─────────────────────────────────────

def get_cached_run_id(cache_key: str) -> int | None:
    with db() as conn:
        row = conn.execute(
            """SELECT run_id FROM result_cache
               WHERE cache_key=? AND (expires_at IS NULL OR expires_at > datetime('now'))""",
            (cache_key,),
        ).fetchone()
        return row["run_id"] if row else None


def set_result_cache(cache_key: str, run_id: int, ttl_days: int = 30) -> None:
    expires = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO result_cache (cache_key, run_id, expires_at) VALUES (?,?,?)",
            (cache_key, run_id, expires),
        )


# ── Rapports ──────────────────────────────────────────────────────────────────

def save_report(run_id: int | None, user_id: int | None,
                report_type: str, file_path: str, params: dict) -> int:
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO reports (run_id, user_id, report_type, file_path, params_json)
               VALUES (?,?,?,?,?)""",
            (run_id, user_id, report_type, file_path, json.dumps(params)),
        )
        return cur.lastrowid


def get_reports(limit: int = 20) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            """SELECT r.id, r.report_type, r.file_path, r.created_at, u.username
               FROM reports r LEFT JOIN users u ON r.user_id=u.id
               ORDER BY r.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Préférences utilisateur ───────────────────────────────────────────────────

def get_pref(user_id: int, key: str, default=None):
    with db() as conn:
        row = conn.execute(
            "SELECT value FROM user_prefs WHERE user_id=? AND key=?", (user_id, key)
        ).fetchone()
        return row["value"] if row else default


def set_pref(user_id: int, key: str, value) -> None:
    with db() as conn:
        conn.execute(
            """INSERT INTO user_prefs (user_id, key, value, updated_at)
               VALUES (?,?,?,datetime('now'))
               ON CONFLICT(user_id, key)
               DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (user_id, key, str(value) if value is not None else None),
        )


def get_all_prefs(user_id: int) -> dict:
    with db() as conn:
        rows = conn.execute(
            "SELECT key, value FROM user_prefs WHERE user_id=?", (user_id,)
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}


# ── Admin Gestion utilisateurs ──────────────────────────────────────────────

def list_users() -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT id, username, role, email, created_at, last_login, is_active FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


def create_user(username: str, password: str, role: str = "analyste") -> int | None:
    """Crée un utilisateur, retourne son id ou None si le username existe déjà."""
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        with db() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                (username, pwd_hash, role),
            )
            return cur.lastrowid
    except Exception:
        return None


def set_user_active(user_id: int, is_active: bool) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE users SET is_active=? WHERE id=?", (1 if is_active else 0, user_id)
        )


def reset_user_password(user_id: int, new_password: str) -> None:
    pwd_hash = hashlib.sha256(new_password.encode()).hexdigest()
    with db() as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?", (pwd_hash, user_id)
        )


def get_audit_log(limit: int = 100, user_id: int | None = None) -> list[dict]:
    query = """SELECT a.id, a.event_time, a.action, a.table_name, a.details_json,
                      u.username
               FROM audit_log a LEFT JOIN users u ON a.user_id=u.id"""
    params: list = []
    if user_id is not None:
        query += " WHERE a.user_id=?"
        params.append(user_id)
    query += " ORDER BY a.event_time DESC LIMIT ?"
    params.append(limit)
    with db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ── Optimisation heuristique ──────────────────────────────────────────────────

def create_optim_run(user_id: int | None, model: str, mode: str,
                     pays: str | None, volet: str | None) -> int:
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO optimization_runs (user_id, model, mode, pays, volet)
               VALUES (?,?,?,?,?)""",
            (user_id, model, mode, pays, volet),
        )
        return cur.lastrowid


def finish_optim_run(run_id: int, status: str, best_params: dict | None,
                     best_score: float | None, n_generations: int | None,
                     duration_ms: int | None) -> None:
    with db() as conn:
        conn.execute(
            """UPDATE optimization_runs
               SET status=?, best_params=?, best_score=?, n_generations=?, duration_ms=?
               WHERE id=?""",
            (status, json.dumps(best_params) if best_params else None,
             best_score, n_generations, duration_ms, run_id),
        )


def get_optim_history(limit: int = 20) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            """SELECT o.id, o.model, o.mode, o.pays, o.volet, o.status,
                      o.best_score, o.n_generations, o.duration_ms, o.created_at,
                      o.best_params, u.username
               FROM optimization_runs o LEFT JOIN users u ON o.user_id=u.id
               ORDER BY o.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Audit log ─────────────────────────────────────────────────────────────────

# ── Paramètres globaux ────────────────────────────────────────────────────────

def get_app_setting(key: str, default=None):
    with db() as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default


def set_app_setting(key: str, value) -> None:
    with db() as conn:
        conn.execute(
            """INSERT INTO app_settings (key, value, updated_at)
               VALUES (?,?,datetime('now'))
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, str(value) if value is not None else None),
        )


# ── Notifications ─────────────────────────────────────────────────────────────

def create_notification_for_all(ntype: str, title: str, body: str,
                                run_id: int | None = None) -> None:
    """Diffuse une notification à tous les utilisateurs actifs (une ligne par destinataire)."""
    with db() as conn:
        user_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM users WHERE is_active=1").fetchall()]
        conn.executemany(
            """INSERT INTO notifications (user_id, run_id, type, title, body)
               VALUES (?,?,?,?,?)""",
            [(uid, run_id, ntype, title, body) for uid in user_ids],
        )


def get_notifications(user_id: int, limit: int = 20) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            """SELECT * FROM notifications WHERE user_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_unread_count(user_id: int) -> int:
    with db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM notifications WHERE user_id=? AND is_read=0",
            (user_id,),
        ).fetchone()
        return row["n"] if row else 0


def mark_notification_read(notif_id: int) -> None:
    with db() as conn:
        conn.execute("UPDATE notifications SET is_read=1 WHERE id=?", (notif_id,))


def mark_all_notifications_read(user_id: int) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE notifications SET is_read=1 WHERE user_id=? AND is_read=0", (user_id,)
        )


def log_audit(user_id: int | None, action: str,
              table_name: str = None, details: dict = None) -> None:
    try:
        with db() as conn:
            conn.execute(
                """INSERT INTO audit_log (user_id, action, table_name, details_json)
                   VALUES (?,?,?,?)""",
                (user_id, action, table_name, json.dumps(details) if details else None),
            )
    except Exception as exc:
        log.warning("audit_log failed: %s", exc)
