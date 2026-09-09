"""Callbacks page Historique — sélection/comparaison de runs, revue des anomalies, commentaires."""
import dash
from dash import callback, Output, Input, State, ctx, html, dcc
from database import models as _db
from pages.historique import _anomaly_table, _comparison_view, _run_row


# ── Sélection d'un run à afficher / comparer ──────────────────────────────────

@callback(
    Output("store-historique-run-a", "data"),
    Input({"type": "hist-view", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_run_a(clicks):
    triggered = ctx.triggered_id
    if triggered is None or not any(c for c in (clicks or []) if c):
        return dash.no_update
    return triggered["index"]


@callback(
    Output("store-historique-run-b", "data"),
    Input({"type": "hist-compare", "index": dash.ALL}, "n_clicks"),
    State("store-historique-run-a", "data"),
    prevent_initial_call=True,
)
def select_run_b(clicks, run_a):
    triggered = ctx.triggered_id
    if triggered is None or not any(c for c in (clicks or []) if c) or not run_a:
        return dash.no_update
    if triggered["index"] == run_a:
        return dash.no_update
    return triggered["index"]


# ── Rendu du panneau droit + rafraîchissement de la liste des runs ───────────

@callback(
    Output("hist-right-panel", "children"),
    Output("hist-run-list",    "children"),
    Input("store-historique-run-a", "data"),
    Input("store-historique-run-b", "data"),
    State("store-hist-runs-cache",  "data"),
    prevent_initial_call=True,
)
def render_right_panel(run_a, run_b, runs_cache):
    runs = runs_cache or _db.get_run_history(limit=50)
    run_list = [_run_row(r, r["id"] == run_a, r["id"] == run_b) for r in runs]

    if run_a and run_b:
        cmp = _db.compare_runs(run_a, run_b)
        return _comparison_view(cmp), run_list

    if run_a:
        anomalies = _db.get_run_anomalies(run_a)
        users = _db.list_users()
        return _anomaly_table(run_a, anomalies, users), run_list

    return dash.no_update, run_list


# ── Actions de revue (marquer revue / faux positif) ───────────────────────────

@callback(
    Output("hist-right-panel", "children", allow_duplicate=True),
    Input({"type": "anomaly-mark-reviewed", "index": dash.ALL}, "n_clicks"),
    Input({"type": "anomaly-mark-fp",       "index": dash.ALL}, "n_clicks"),
    State("store-historique-run-a", "data"),
    State("store-auth",             "data"),
    prevent_initial_call=True,
)
def mark_anomaly_status(clicks_rev, clicks_fp, run_a, auth):
    triggered = ctx.triggered_id
    if triggered is None or not run_a:
        return dash.no_update
    all_clicks = (clicks_rev or []) + (clicks_fp or [])
    if not any(c for c in all_clicks if c):
        return dash.no_update

    status = "reviewed" if triggered["type"] == "anomaly-mark-reviewed" else "false_positive"
    reviewer_id = (auth or {}).get("user_id")
    _db.update_anomaly_status(triggered["index"], status, reviewer_id)
    _db.log_audit(reviewer_id, "anomaly_status_updated",
                  details={"anomaly_id": triggered["index"], "status": status})

    anomalies = _db.get_run_anomalies(run_a)
    users = _db.list_users()
    return _anomaly_table(run_a, anomalies, users)


@callback(
    Output("hist-right-panel", "children", allow_duplicate=True),
    Input({"type": "anomaly-assign", "index": dash.ALL}, "value"),
    State("store-historique-run-a", "data"),
    prevent_initial_call=True,
)
def assign_anomaly(values, run_a):
    triggered = ctx.triggered_id
    if triggered is None or not run_a:
        return dash.no_update
    assignee = triggered and None
    idx = triggered["index"]
    # Retrouve la valeur choisie pour cette ligne précise via l'id déclencheur
    for v, ident in zip(values, ctx.inputs_list[0]):
        if ident["id"]["index"] == idx:
            assignee = v or None
            break
    _db.assign_anomaly(idx, assignee)
    anomalies = _db.get_run_anomalies(run_a)
    users = _db.list_users()
    return _anomaly_table(run_a, anomalies, users)


# ── Commentaires (modal générique) ────────────────────────────────────────────

def _comment_modal_body(anomaly_id: int) -> html.Div:
    comments = _db.get_anomaly_comments(anomaly_id)
    items = [
        html.Div(style={"marginBottom": "10px", "paddingBottom": "8px",
                        "borderBottom": "1px solid var(--border)"}, children=[
            html.Div([
                html.Span(c.get("username") or "—", style={"fontWeight": "600", "fontSize": "11px"}),
                html.Span((c.get("created_at") or "")[:16],
                          style={"fontSize": "10px", "color": "var(--muted2)", "marginLeft": "8px"}),
            ]),
            html.Div(c["body"], style={"fontSize": "12px", "marginTop": "2px"}),
        ])
        for c in comments
    ] or [html.Div("Aucun commentaire pour l'instant.",
                   style={"fontSize": "11px", "color": "var(--muted2)", "marginBottom": "10px"})]

    return html.Div([
        html.Div(id="modal-comments-list", children=items),
        html.Div(className="login-field-wrap", style={"marginTop": "10px"}, children=[
            dcc.Textarea(id="inp-modal-comment", placeholder="Ajouter un commentaire…",
                        className="input-field", style={"width": "100%", "minHeight": "60px"}),
        ]),
        html.Button("Ajouter", id="btn-modal-add-comment", n_clicks=0,
                    className="btn btn-gold", style={"marginTop": "8px"}),
    ])


@callback(
    Output("app-modal-overlay",     "className", allow_duplicate=True),
    Output("app-modal-title",       "children",  allow_duplicate=True),
    Output("app-modal-body",        "children",  allow_duplicate=True),
    Output("app-modal-footer",      "children",  allow_duplicate=True),
    Output("store-modal-anomaly-id", "data"),
    Input({"type": "anomaly-comment-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def open_comment_modal(clicks):
    triggered = ctx.triggered_id
    if triggered is None or not any(c for c in (clicks or []) if c):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    anomaly_id = triggered["index"]
    return "open", "Commentaires", _comment_modal_body(anomaly_id), [], anomaly_id


@callback(
    Output("app-modal-body", "children", allow_duplicate=True),
    Output("hist-right-panel", "children", allow_duplicate=True),
    Input("btn-modal-add-comment",   "n_clicks"),
    State("inp-modal-comment",       "value"),
    State("store-modal-anomaly-id",  "data"),
    State("store-historique-run-a",  "data"),
    State("store-auth",              "data"),
    prevent_initial_call=True,
)
def add_comment(n, body, anomaly_id, run_a, auth):
    if not n or not body or not (body or "").strip() or not anomaly_id:
        return dash.no_update, dash.no_update
    _db.add_anomaly_comment(anomaly_id, (auth or {}).get("user_id"), body.strip())
    refreshed_table = dash.no_update
    if run_a:
        anomalies = _db.get_run_anomalies(run_a)
        users = _db.list_users()
        refreshed_table = _anomaly_table(run_a, anomalies, users)
    return _comment_modal_body(anomaly_id), refreshed_table
