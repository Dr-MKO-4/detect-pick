"""Callback d'authentification — utilise SQLite via database.models."""
import hashlib
import dash
from dash import callback, Output, Input, State
from database import models
from database.models import log_audit


@callback(
    Output("store-page",   "data", allow_duplicate=True),
    Output("store-auth",   "data"),
    Output("login-error",  "children"),
    Output("store-theme",  "data", allow_duplicate=True),
    Output("store-pays",   "data", allow_duplicate=True),
    Output("store-volet",  "data", allow_duplicate=True),
    Output("store-modele", "data", allow_duplicate=True),
    Input("btn-login",     "n_clicks"),
    State("inp-login-user", "value"),
    State("inp-login-pwd",  "value"),
    prevent_initial_call=True,
)
def do_login(n_clicks, user, pwd):
    _no = dash.no_update
    if not n_clicks:
        return _no, _no, "", _no, _no, _no, _no
    if not user:
        return _no, _no, "Identifiant requis.", _no, _no, _no, _no
    if not pwd:
        return _no, _no, "Mot de passe requis.", _no, _no, _no, _no

    username = (user or "").strip().lower()
    db_user  = models.get_user(username)

    if db_user is None:
        log_audit(None, "login_failed", details={"username": username})
        return _no, _no, "Identifiant ou mot de passe incorrect.", _no, _no, _no, _no

    entered = hashlib.sha256((pwd or "").encode()).hexdigest()
    if entered != db_user["password_hash"]:
        log_audit(db_user["id"], "login_failed", details={"username": username})
        return _no, _no, "Identifiant ou mot de passe incorrect.", _no, _no, _no, _no

    token = models.create_session(db_user["id"])
    log_audit(db_user["id"], "login_success", details={"username": username})

    import datetime
    auth_payload = {
        "user":     db_user["username"],
        "role":     db_user["role"],
        "user_id":  db_user["id"],
        "token":    token,
        "login_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Load saved preferences (with defaults)
    prefs = models.get_all_prefs(db_user["id"])
    theme  = prefs.get("theme",  "light")
    pays   = prefs.get("pays",   "cameroun")
    volet  = prefs.get("volet",  "Actif")
    modele = prefs.get("modele", "lof")

    return "donnees", auth_payload, "", theme, pays, volet, modele


@callback(
    Output("store-page", "data", allow_duplicate=True),
    Output("store-auth", "data", allow_duplicate=True),
    Input("btn-logout",  "n_clicks"),
    State("store-auth",  "data"),
    prevent_initial_call=True,
)
def do_logout(n, auth):
    if not n:
        return dash.no_update, dash.no_update
    if auth:
        models.close_session(auth.get("token"))
        log_audit(auth.get("user_id"), "logout", details={"username": auth.get("user")})
    return "login", None
