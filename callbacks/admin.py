"""Callbacks page Admin — gestion utilisateurs + audit."""
import secrets
import string
import dash
from dash import callback, Output, Input, State, ctx, html
from database import models as _db
from pages.admin import _users_tab, _create_tab, _audit_tab, _system_tab


# ── Rendu des onglets ─────────────────────────────────────────────────────────

@callback(
    Output("admin-tab-content", "children"),
    Input("admin-tabs",          "value"),
    State("store-admin-users",   "data"),
    State("store-admin-audit",   "data"),
)
def render_admin_tab(tab, users, audit):
    if tab == "users":
        return _users_tab(users or [])
    if tab == "create":
        return _create_tab()
    if tab == "audit":
        return _audit_tab(audit or [])
    if tab == "system":
        return _system_tab()
    return dash.no_update


@callback(
    Output("alert-threshold-msg",      "children"),
    Output("alert-threshold-msg",      "style"),
    Input("btn-save-alert-threshold",  "n_clicks"),
    State("inp-alert-threshold",       "value"),
    State("store-auth",                "data"),
    prevent_initial_call=True,
)
def save_alert_threshold(n, value, auth):
    if not n:
        return dash.no_update, dash.no_update
    try:
        threshold = max(1, int(value))
    except (TypeError, ValueError):
        return "Valeur invalide.", {"color": "var(--red)", "fontSize": "11px"}
    _db.set_app_setting("alert_threshold_anomalies", threshold)
    _db.log_audit((auth or {}).get("user_id"), "alert_threshold_updated",
                  details={"threshold": threshold})
    return (f"Seuil enregistré : {threshold} anomalies.",
            {"color": "var(--green)", "fontSize": "11px"})


# ── Déclencher la boîte de confirmation ──────────────────────────────────────
# store-admin-pending : {"type": "toggle"|"reset", "user_id": int}

@callback(
    Output("store-admin-pending",  "data"),
    Output("app-modal-overlay",    "className", allow_duplicate=True),
    Output("app-modal-title",      "children",  allow_duplicate=True),
    Output("app-modal-body",       "children",  allow_duplicate=True),
    Output("app-modal-footer",     "children",  allow_duplicate=True),
    Input({"type": "btn-toggle-user", "index": dash.ALL}, "n_clicks"),
    Input({"type": "btn-reset-pwd",   "index": dash.ALL}, "n_clicks"),
    State("store-admin-users", "data"),
    prevent_initial_call=True,
)
def ask_admin_action(toggle_clicks, reset_clicks, users):
    triggered = ctx.triggered_id
    if triggered is None or not any(c for c in (list(toggle_clicks or []) + list(reset_clicks or [])) if c):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    action_type = triggered["type"]   # "btn-toggle-user" or "btn-reset-pwd"
    user_id     = triggered["index"]
    user = next((u for u in (users or []) if u["id"] == user_id), None)
    if user is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    footer = [
        html.Button("Annuler", id="app-modal-cancel", className="btn btn-ghost", n_clicks=0),
        html.Button("Confirmer", id="app-modal-confirm-btn-adminaction",
                    className="btn btn-red", n_clicks=0),
    ]

    if action_type == "btn-toggle-user":
        verb = "désactiver" if user["is_active"] else "réactiver"
        msg  = f"Voulez-vous {verb} le compte « {user['username']} » ?"
        return ({"type": "toggle", "user_id": user_id}, "open",
                "Gestion du compte", html.Div(msg), footer)

    verb = "réinitialiser"
    msg  = (f"Voulez-vous {verb} le mot de passe de « {user['username']} » ? "
            "Un mot de passe temporaire sera affiché — à transmettre manuellement.")
    return ({"type": "reset", "user_id": user_id}, "open",
            "Gestion du compte", html.Div(msg), footer)


# ── Exécuter l'action confirmée ───────────────────────────────────────────────

@callback(
    Output("store-admin-users", "data"),
    Output("admin-user-msg",    "children"),
    Output("admin-user-msg",    "style"),
    Output("app-modal-overlay", "className", allow_duplicate=True),
    Input("app-modal-confirm-btn-adminaction", "n_clicks"),
    State("store-admin-pending",    "data"),
    State("store-admin-users",      "data"),
    State("store-auth",             "data"),
    prevent_initial_call=True,
)
def do_admin_action(n, pending, users, auth):
    if not n or not pending:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    action_type = pending.get("type")
    user_id     = pending.get("user_id")
    user = next((u for u in (users or []) if u["id"] == user_id), None)
    if user is None:
        return (dash.no_update, "Utilisateur introuvable.",
                {"color": "var(--red)", "fontSize": "11px"}, "")

    admin_id = (auth or {}).get("user_id")

    if action_type == "toggle":
        new_active = not bool(user["is_active"])
        _db.set_user_active(user_id, new_active)
        _db.log_audit(admin_id, "user_toggled",
                      details={"target_user_id": user_id, "is_active": new_active})
        label  = "réactivé" if new_active else "désactivé"
        color  = "var(--green)" if new_active else "var(--red)"
        updated = _db.list_users()
        return (updated,
                f"Compte {user['username']} {label}.",
                {"marginTop": "8px", "fontSize": "11px", "color": color}, "")

    if action_type == "reset":
        alphabet = string.ascii_letters + string.digits
        new_pwd  = "".join(secrets.choice(alphabet) for _ in range(12))
        _db.reset_user_password(user_id, new_pwd)
        _db.log_audit(admin_id, "password_reset",
                      details={"target_user_id": user_id, "username": user["username"]})
        updated = _db.list_users()
        return (updated,
                f"MDP de {user['username']} réinitialisé. Temporaire : {new_pwd}",
                {"marginTop": "8px", "fontSize": "11px", "color": "var(--gold)"}, "")

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


# ── Créer un utilisateur ──────────────────────────────────────────────────────

@callback(
    Output("create-user-msg",    "children"),
    Output("create-user-msg",    "style"),
    Output("inp-new-username",   "value"),
    Output("inp-new-password",   "value"),
    Input("btn-create-user",     "n_clicks"),
    State("inp-new-username",    "value"),
    State("inp-new-password",    "value"),
    State("dd-new-role",         "value"),
    State("store-auth",          "data"),
    prevent_initial_call=True,
)
def create_user(n, username, password, role, auth):
    if not n:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    username = (username or "").strip()
    password = password or ""
    if not username:
        return ("Nom d'utilisateur requis.",
                {"fontSize": "11px", "color": "var(--red)"},
                dash.no_update, dash.no_update)
    if len(password) < 6:
        return ("Mot de passe trop court (minimum 6 caractères).",
                {"fontSize": "11px", "color": "var(--red)"},
                dash.no_update, dash.no_update)
    new_id = _db.create_user(username, password, role)
    if new_id is None:
        return (f"Nom d'utilisateur « {username} » déjà utilisé.",
                {"fontSize": "11px", "color": "var(--red)"},
                dash.no_update, dash.no_update)
    _db.log_audit(
        (auth or {}).get("user_id"),
        "user_created",
        details={"new_user": username, "role": role},
    )
    return (f"Compte « {username} » créé avec le rôle {role}.",
            {"fontSize": "11px", "color": "var(--green)"},
            "", "")
