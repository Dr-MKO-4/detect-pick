"""Persist user preferences to SQLite whenever relevant stores change."""
import dash
from dash import callback, Output, Input, State
from database import models


@callback(
    Output("store-pref-sync", "data"),
    Input("store-theme",  "data"),
    Input("store-pays",   "data"),
    Input("store-volet",  "data"),
    Input("store-modele", "data"),
    State("store-auth",   "data"),
    prevent_initial_call=True,
)
def save_prefs(theme, pays, volet, modele, auth):
    if not auth:
        return dash.no_update
    uid = auth.get("user_id")
    if not uid:
        return dash.no_update
    for key, val in [("theme", theme), ("pays", pays), ("volet", volet), ("modele", modele)]:
        if val is not None:
            models.set_pref(uid, key, val)
    return dash.no_update
