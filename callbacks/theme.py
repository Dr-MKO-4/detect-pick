"""Callbacks thème clair/sombre. Thème clair = défaut (:root). Sombre = [data-theme="dark"]."""
import dash
from dash import callback, clientside_callback, Output, Input, State


clientside_callback(
    """
    function(theme) {
        if (theme === 'dark') {
            document.documentElement.dataset.theme = 'dark';
        } else {
            delete document.documentElement.dataset.theme;
        }
        return theme || 'light';
    }
    """,
    Output("btn-theme",  "children"),
    Input("store-theme", "data"),
)


clientside_callback(
    """
    function(n_clicks, current_theme) {
        if (!n_clicks) return window.dash_clientside.no_update;
        return current_theme === 'dark' ? 'light' : 'dark';
    }
    """,
    Output("store-theme", "data"),
    Input("btn-theme",    "n_clicks"),
    State("store-theme",  "data"),
    prevent_initial_call=True,
)


@callback(
    Output("store-theme", "data", allow_duplicate=True),
    Input("btn-theme-toggle", "n_clicks"),
    State("store-theme",      "data"),
    prevent_initial_call=True,
)
def toggle_theme_qt(n, current):
    return "dark" if current == "light" else "light"
