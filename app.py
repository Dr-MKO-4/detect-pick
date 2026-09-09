"""BEAC Anomaly Detector  point d'entrée Dash.

Dev  : python app.py          (BEAC_DEV=1 → hot-reload)
Prod : importé par main.py    (BEAC_DEV=0, PyQt6 + QWebEngineView)
"""
import os
import logging
import logging.handlers

# ── Logging ────────────────────────────────────────────────────────────────────
_LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
_fh = logging.handlers.RotatingFileHandler(
    os.path.join(_LOG_DIR, "beac.log"), maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
_fh.setFormatter(_fmt)
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_fh, _ch])

# ── Dev/prod switch ────────────────────────────────────────────────────────────
DEV = os.getenv("BEAC_DEV", "1") == "1"

import dash
from dash import Dash, html, dcc

# ── App Dash ───────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="BEAC · Détection d'anomalies",
    update_title=None,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)
server = app.server  # exposition WSGI pour Waitress/Gunicorn

# ── Layout racine ──────────────────────────────────────────────────────────────
app.layout = html.Div(
    id="app-root",
    children=[
        # Stores d'état
        dcc.Store(id="store-page",             data="login"),
        dcc.Store(id="store-theme",            data="light"),  # clair = défaut
        dcc.Store(id="store-auth",             data=None),
        dcc.Store(id="store-results",          data=None),
        dcc.Store(id="store-bivat-results",    data=None),
        dcc.Store(id="store-pipeline-running", data=False),
        dcc.Store(id="store-bivat-running",    data=False),
        dcc.Store(id="store-modele",           data="lof"),
        dcc.Store(id="store-phase",            data=1),
        dcc.Store(id="store-pays",             data="cameroun"),
        dcc.Store(id="store-volet",            data="Actif"),
        dcc.Store(id="store-annee",            data="all"),
        dcc.Store(id="store-pref-sync",        data=None),
        dcc.Store(id="store-progress",         data=0),
        dcc.Store(id="store-progress-label",   data=""),
        dcc.Store(id="store-notif-count",      data=0),
        dcc.Store(id="store-refresh",          data=0),
        dcc.Store(id="store-historique-run-a", data=None),
        dcc.Store(id="store-historique-run-b", data=None),
        dcc.Store(id="store-donnees-del-idx",  data=None),
        dcc.Store(id="store-modal-anomaly-id", data=None),
        # Intervalles de polling pipeline + optimisation
        dcc.Interval(id="interval-pipeline",    interval=2_000, n_intervals=0, disabled=True),
        dcc.Interval(id="interval-bivat",       interval=2_000, n_intervals=0, disabled=True),
        dcc.Interval(id="interval-lof-optim",   interval=2_000, n_intervals=0, disabled=True),
        dcc.Interval(id="interval-bivat-optim", interval=3_000, n_intervals=0, disabled=True),
        dcc.Interval(id="interval-notif",       interval=15_000, n_intervals=0, disabled=False),
        # Bouton invisible : déclenché par la barre de menus Qt pour basculer le thème
        html.Button(id="btn-theme-toggle", style={"display": "none"}, n_clicks=0),
        # Cible du callback clientside theme.py — applique data-theme au DOM
        html.Span(id="btn-theme", style={"display": "none"}),
        html.Span(id="pg-dummy", style={"display": "none"}),
        html.Span(id="global-anim-dummy", style={"display": "none"}),
        html.Span(id="annee-dummy", style={"display": "none"}),
        # ── Animations globales (persistent across page navigation) ───────────
        # Barre fine dorée fixée en haut de l'écran
        html.Div(id="global-top-bar"),
        # Toast flottant bas-droite avec spinner + label + barre de progression
        html.Div(id="global-status-toast", children=[
            html.Div(className="gst-inner", children=[
                html.Div(className="global-spinner"),
                html.Div(className="gst-text", children=[
                    html.Div(id="gst-title", className="gst-title", children=""),
                    html.Div(id="gst-label", className="gst-label", children=""),
                ]),
                html.Div(id="gst-pct", className="gst-pct", children=""),
            ]),
            html.Div(className="gst-bar-track", children=[
                html.Div(id="gst-bar-fill", className="gst-bar-fill"),
            ]),
        ]),
        # Conteneur de page (login OU shell + contenu)
        html.Div(id="page-container",
                 style={"width": "100%", "height": "100vh", "overflow": "hidden"}),

        # ── Modal générique (remplace les confirm() natifs du navigateur) ─────
        html.Div(id="app-modal-overlay", className="", children=[
            html.Div(id="app-modal", role="dialog", **{"aria-modal": "true"}, children=[
                html.Div(className="app-modal-header", children=[
                    html.Span(id="app-modal-title", className="app-modal-title"),
                    html.Span(id="app-modal-close", className="app-modal-close",
                             role="button", tabIndex=0,
                             children=[html.Span("×", style={"fontSize": "20px", "lineHeight": "1"})]),
                ]),
                html.Div(id="app-modal-body", className="app-modal-body"),
                html.Div(id="app-modal-footer", className="app-modal-footer"),
            ]),
        ]),

        # ── Panneau de notifications (déclenché par la cloche sidebar) ────────
        html.Div(id="notif-panel", className="", children=[
            html.Div(id="notif-panel-list"),
        ]),

        # ── Palette de recherche globale (Ctrl+K) ──────────────────────────────
        html.Div(id="search-palette-overlay", className="", children=[
            html.Div(id="search-palette", children=[
                html.Div(className="search-input-row", children=[
                    dcc.Input(id="inp-search-query", type="text",
                              placeholder="Rechercher une page, un pays, un volet, un fichier…",
                              debounce=True, autoComplete="off"),
                ]),
                html.Div(id="search-results-list"),
            ]),
        ]),
    ],
)

# ── Initialisation BD (une seule fois) ────────────────────────────────────────
from database import init_db
init_db()

# ── Enregistrement des callbacks (ordre figé) ──────────────────────────────────
import callbacks  # noqa: F401,E402   importe auth, navigation, theme, pipeline, extras


# ── Callback principal de rendu de page ───────────────────────────────────────
from dash import callback, Output, Input, State, ctx
from pages  import (render_login, donnees_layout, scan_files,
                    scan_files_cached, refresh_scan_cache,
                    render_analyse, render_rapport, render_modeles,
                    render_parametres, render_compte,
                    render_admin, render_optimisation)
from database import models as _db
from layout import shell_layout


@callback(
    Output("page-container", "children"),
    Input("store-page",             "data"),
    Input("store-results",          "data"),
    Input("store-bivat-results",    "data"),
    Input("store-phase",            "data"),
    Input("store-pipeline-running", "data"),
    Input("store-bivat-running",    "data"),
    Input("store-modele",           "data"),
    Input("store-refresh",          "data"),
    State("store-progress",         "data"),
    State("store-progress-label",   "data"),
    State("store-pays",             "data"),
    State("store-volet",            "data"),
    State("store-annee",            "data"),
    State("store-auth",             "data"),
    State("store-theme",            "data"),
)
def render_page(page, results, bivat_results, phase,
                running, bivat_running, modele, _refresh, progress, progress_label,
                pays, volet, annee, auth, theme):
    # Rediriger vers login si pas authentifié
    if not auth and page != "login":
        return render_login()

    any_running = bool(running) or bool(bivat_running)
    username    = (auth or {}).get("user", "analyste.beac")
    role        = (auth or {}).get("role", "analyste")
    user_id     = (auth or {}).get("user_id")
    unread      = _db.get_unread_count(user_id) if user_id else 0

    def _shell(p, content, **kw):
        return shell_layout(p, content, username=username, role=role,
                            unread_count=unread, **kw)

    if page == "login":
        return render_login()

    if page == "donnees":
        # Ne re-scanne le dossier data/ que si on vient d'y naviguer — un store
        # sans rapport (progression pipeline, résultats…) ne doit pas relancer
        # le glob + la lecture des métadonnées XLSX.
        files = (refresh_scan_cache()
                 if ctx.triggered_id in ("store-page", "store-refresh")
                 else scan_files_cached())
        return _shell(page, donnees_layout(files),
                      pays=pays or "cameroun", volet=volet or "Actif",
                      running=any_running)

    if page == "analyse":
        content = render_analyse(
            pays=pays or "cameroun", volet=volet or "Actif",
            modele=modele or "lof", phase=phase or 1,
            running=any_running, results=results, bivat_results=bivat_results,
            progress=progress or 0, progress_label=progress_label or "",
            annee=annee or "all",
        )
        return _shell(page, content, running=any_running,
                      pays=pays or "cameroun", volet=volet or "Actif")

    if page == "rapport":
        reports = _db.get_reports(limit=20)
        return _shell(page, render_rapport(reports), running=any_running,
                      pays=pays or "cameroun", volet=volet or "Actif")

    if page == "modeles":
        return _shell(page, render_modeles(), running=any_running,
                      pays=pays or "cameroun", volet=volet or "Actif")

    if page == "optimisation":
        return _shell(page, render_optimisation(), running=any_running,
                      pays=pays or "cameroun", volet=volet or "Actif")

    if page == "parametres":
        return _shell(page, render_parametres(theme=theme or "light"), running=any_running,
                      pays=pays or "cameroun", volet=volet or "Actif")

    if page == "compte":
        return _shell(page, render_compte(auth), running=any_running,
                      pays=pays or "cameroun", volet=volet or "Actif")

    if page == "admin":
        if role != "admin":
            return render_login()
        return _shell(page, render_admin(auth), running=any_running,
                      pays=pays or "cameroun", volet=volet or "Actif")

    if page == "historique":
        from pages import render_historique
        return _shell(page, render_historique(), running=any_running,
                      pays=pays or "cameroun", volet=volet or "Actif")

    return render_login()


# ── Lancement en mode développement ───────────────────────────────────────────
if __name__ == "__main__":
    _HOST = "127.0.0.1"
    _PORT = 8050

    app.run(debug=DEV, host=_HOST, port=_PORT, use_reloader=False)
