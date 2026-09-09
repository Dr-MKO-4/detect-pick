"""Callbacks de navigation entre pages et phases."""
import dash
from dash import callback, Output, Input, State


@callback(
    Output("store-page", "data"),
    Input("nav-donnees",      "n_clicks"),
    Input("nav-analyse",      "n_clicks"),
    Input("nav-rapport",      "n_clicks"),
    Input("nav-modeles",      "n_clicks"),
    Input("nav-optimisation", "n_clicks"),
    Input("nav-historique",   "n_clicks"),
    Input("nav-parametres",   "n_clicks"),
    Input("nav-compte",       "n_clicks"),
    Input("nav-admin",        "n_clicks"),
    State("store-page",  "data"),
    State("store-auth",  "data"),
    prevent_initial_call=True,
)
def navigate(n_d, n_a, n_r, n_m, n_o, n_h, n_p, n_c, n_adm, current_page, auth):
    if not auth:
        return "login"
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_page
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    return {
        "nav-donnees":      "donnees",
        "nav-analyse":      "analyse",
        "nav-rapport":      "rapport",
        "nav-modeles":      "modeles",
        "nav-optimisation": "optimisation",
        "nav-historique":   "historique",
        "nav-parametres":   "parametres",
        "nav-compte":       "compte",
        "nav-admin":        "admin",
    }.get(trigger, current_page)


@callback(
    Output("store-phase", "data"),
    Input("btn-phase-1",  "n_clicks"),
    Input("btn-phase-2",  "n_clicks"),
    Input("btn-phase-3",  "n_clicks"),
    Input("btn-phase-4",  "n_clicks"),
    State("store-phase",  "data"),
    prevent_initial_call=True,
)
def update_phase(n1, n2, n3, n4, current_phase):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_phase
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    return {"btn-phase-1": 1, "btn-phase-2": 2,
            "btn-phase-3": 3, "btn-phase-4": 4}.get(trigger, current_phase)


@callback(
    Output("store-pays",  "data"),
    Output("store-volet", "data"),
    Input("dd-pays",  "value"),
    Input("dd-volet", "value"),
    prevent_initial_call=True,
)
def sync_pays_volet(pays, volet):
    return pays or "cameroun", volet or "Actif"


@callback(
    Output("store-modele", "data", allow_duplicate=True),
    Output("dd-modele",    "value"),
    Input("btn-model-lof",   "n_clicks"),
    Input("btn-model-bivat", "n_clicks"),
    prevent_initial_call=True,
)
def switch_model_btn(n_lof, n_bivat):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    val = "lof" if trigger == "btn-model-lof" else "bivat"
    return val, val


@callback(
    Output("store-modele", "data"),
    Input("dd-modele",     "value"),
    prevent_initial_call=True,
)
def sync_modele(v):
    return v or "lof"
