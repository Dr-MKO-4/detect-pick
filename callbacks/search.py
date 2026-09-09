"""Callbacks — palette de recherche globale (Ctrl+K).

Portée volontairement limitée : pages accessibles selon le rôle, pays, volets,
fichiers XLSX du dossier data/. Pas d'indexation du contenu des fichiers
(indicateurs) — hors scope de cette passe.
"""
import dash
from dash import callback, Output, Input, State, ctx, html
from config import PAYS, PAYS_LABELS, VOLETS
from layout.icons import svg, ICO_FILE, ICO_PULSE, ICO_GRID, ICO_MODEL, ICO_SLIDERS, ICO_CLOCK, ICO_SETTINGS, ICO_USERS

_PAGES = [
    ("donnees",      "Données",      ICO_FILE,     None),
    ("analyse",      "Analyse",      ICO_PULSE,    None),
    ("rapport",      "Rapport",      ICO_GRID,     None),
    ("modeles",      "Modèles",      ICO_MODEL,    None),
    ("optimisation", "Optimisation", ICO_SLIDERS,  None),
    ("historique",   "Historique",   ICO_CLOCK,    None),
    ("parametres",   "Paramètres",   ICO_SETTINGS, None),
    ("admin",        "Administration", ICO_USERS,  "admin"),
]


def _result_button(icon, label, sub, kind, value):
    return html.Button(
        [svg(icon, size=13, stroke="var(--gold)"),
         html.Span(label, style={"marginLeft": "4px"}),
         html.Span(sub, className="search-result-type")],
        id={"type": "search-result", "kind": kind, "value": str(value)},
        n_clicks=0, className="search-result-item",
    )


@callback(
    Output("search-results-list", "children"),
    Input("inp-search-query", "value"),
    State("store-auth",       "data"),
    prevent_initial_call=True,
)
def filter_search(query, auth):
    role = (auth or {}).get("role", "analyste")
    q = (query or "").strip().lower()

    results = []
    for page_id, label, icon, req_role in _PAGES:
        if req_role and role != req_role:
            continue
        if not q or q in label.lower() or q in page_id:
            results.append(_result_button(icon, label, "page", "page", page_id))

    for code in PAYS:
        label = PAYS_LABELS.get(code, code)
        if not q or q in label.lower() or q in code:
            results.append(_result_button(ICO_GRID, label, "pays", "pays", code))

    for volet in VOLETS:
        if not q or q in volet.lower():
            results.append(_result_button(ICO_GRID, volet, "volet", "volet", volet))

    if q:
        try:
            from pages.donnees import scan_files
            for f in scan_files():
                if f.get("ok") and q in f["name"].lower():
                    results.append(_result_button(ICO_FILE, f["name"], "fichier", "fichier", f["name"]))
        except Exception:
            pass

    if not results:
        return html.Div("Aucun résultat.",
                        style={"padding": "16px", "fontSize": "12px", "color": "var(--muted2)"})
    return results[:30]


@callback(
    Output("store-page",              "data", allow_duplicate=True),
    Output("store-pays",              "data", allow_duplicate=True),
    Output("store-volet",             "data", allow_duplicate=True),
    Output("search-palette-overlay",  "className"),
    Output("inp-search-query",        "value"),
    Input({"type": "search-result", "kind": dash.ALL, "value": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def go_to_result(clicks):
    triggered = ctx.triggered_id
    if triggered is None or not any(c for c in (clicks or []) if c):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    kind, value = triggered["kind"], triggered["value"]
    page_out = dash.no_update
    pays_out = dash.no_update
    volet_out = dash.no_update

    if kind == "page":
        page_out = value
    elif kind == "pays":
        page_out, pays_out = "analyse", value
    elif kind == "volet":
        page_out, volet_out = "analyse", value
    elif kind == "fichier":
        page_out = "donnees"

    return page_out, pays_out, volet_out, "", ""
