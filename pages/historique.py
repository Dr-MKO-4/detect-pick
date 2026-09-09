"""Page Historique  runs passés, comparaison, revue des anomalies détectées."""
from dash import html, dcc
from layout.icons import svg, ICO_CLOCK, ICO_GRID, ICO_MSG

_STATUS_LABELS = {
    "new": ("Nouvelle", "grey"),
    "reviewed": ("Revue", "green"),
    "false_positive": ("Faux positif", "gold"),
}


def _run_row(run: dict, is_a: bool, is_b: bool) -> html.Div:
    cls = "settings-item"
    if is_a or is_b:
        cls += " active"
    status_color = "green" if run.get("status") == "success" else "red" if run.get("status") == "error" else "grey"
    return html.Div(className=cls, style={"flexDirection": "column", "alignItems": "stretch", "gap": "4px",
                                          "padding": "8px 12px", "borderLeft": "2px solid transparent"},
                    children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
            html.Span((run.get("created_at") or "")[:16], style={"fontSize": "10px", "fontFamily": "var(--font-mono)",
                                                                   "color": "var(--muted)"}),
            html.Span(className=f"tag tag-{status_color}", children=run.get("status", "—")),
        ]),
        html.Div(f"{run.get('pays','')} · {run.get('volet','')} · {(run.get('modele') or '').upper()}",
                 style={"fontSize": "12px", "fontWeight": "500"}),
        html.Div(f"{run.get('n_anomalies', '—')} anomalies · τ={run.get('tau_mean') or '—'} · "
                 f"{run.get('duration_ms', 0) or 0} ms",
                 style={"fontSize": "10px", "color": "var(--muted2)"}),
        html.Div(style={"display": "flex", "gap": "6px", "marginTop": "2px"}, children=[
            html.Button("Voir", id={"type": "hist-view", "index": run["id"]}, n_clicks=0,
                        className="btn btn-sm btn-gold" if is_a else "btn btn-sm btn-ghost"),
            html.Button("Comparer", id={"type": "hist-compare", "index": run["id"]}, n_clicks=0,
                        className="btn btn-sm btn-gold" if is_b else "btn btn-sm btn-ghost"),
        ]),
    ])


def _anomaly_table(run_id: int, anomalies: list[dict], users: list[dict]) -> html.Div:
    if not anomalies:
        return html.Div("Aucune anomalie enregistrée pour ce run.",
                        style={"padding": "20px", "color": "var(--muted2)", "fontSize": "12px"})

    user_options = [{"label": "Non assigné", "value": ""}] + \
        [{"label": u["username"], "value": u["id"]} for u in users]

    rows = []
    for a in anomalies:
        status_label, status_color = _STATUS_LABELS.get(a.get("status") or "new", ("Nouvelle", "grey"))
        is_anom = bool(a.get("is_anomaly"))
        rows.append(html.Tr([
            html.Td(a.get("period", "—"), style={"fontFamily": "var(--font-mono)"}),
            html.Td("Anomalie" if is_anom else "Normal",
                    className="anomaly" if is_anom else "normal",
                    style={"fontWeight": "700" if is_anom else "400"}),
            html.Td(f"{a.get('lof_score'):.3f}" if a.get("lof_score") is not None else "—",
                    style={"fontFamily": "var(--font-mono)"}),
            html.Td(html.Span(status_label, className=f"tag tag-{status_color}")),
            html.Td(dcc.Dropdown(
                id={"type": "anomaly-assign", "index": a["id"]},
                options=user_options, value=a.get("assignee_user_id") or "",
                clearable=False, className="dd-input", style={"minWidth": "140px", "fontSize": "11px"},
            )),
            html.Td([
                html.Button("Revue", id={"type": "anomaly-mark-reviewed", "index": a["id"]},
                            n_clicks=0, className="btn btn-sm btn-ghost", style={"marginRight": "4px"}),
                html.Button("Faux +", id={"type": "anomaly-mark-fp", "index": a["id"]},
                            n_clicks=0, className="btn btn-sm btn-ghost", style={"marginRight": "4px"}),
                html.Button(
                    [svg(ICO_MSG, size=10), f" {a.get('n_comments', 0)}"],
                    id={"type": "anomaly-comment-btn", "index": a["id"]},
                    n_clicks=0, className="btn btn-sm btn-ghost",
                    **{"aria-label": f"Commentaires — période {a.get('period','')}"},
                ),
            ], style={"whiteSpace": "nowrap"}),
        ]))

    return html.Table(className="data-table", style={"width": "100%"}, children=[
        html.Thead(html.Tr([
            html.Th("Période"), html.Th("Statut détection"), html.Th("Score LOF*"),
            html.Th("Revue"), html.Th("Assigné à"), html.Th("Actions"),
        ])),
        html.Tbody(rows, id="hist-anomaly-tbody"),
    ])


def _comparison_view(cmp: dict) -> html.Div:
    ra, rb = cmp["run_a"], cmp["run_b"]

    def _card(run):
        if not run:
            return html.Div("—", className="stat-card")
        return html.Div(className="panel", style={"margin": "0", "flex": "1"}, children=[
            html.Div(className="panel-header", children=[
                html.Span(f"Run #{run['id']}", className="panel-title"),
            ]),
            html.Div(className="panel-body", children=[
                html.Div(f"{run.get('pays','')} · {run.get('volet','')} · {(run.get('modele') or '').upper()}",
                         style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "6px"}),
                html.Div(f"Anomalies : {run.get('n_anomalies','—')}", style={"fontSize": "11px"}),
                html.Div(f"τ moyen : {run.get('tau_mean','—')}", style={"fontSize": "11px"}),
                html.Div(f"Durée : {run.get('duration_ms','—')} ms", style={"fontSize": "11px"}),
                html.Div(f"Le {(run.get('created_at') or '')[:16]}",
                         style={"fontSize": "10px", "color": "var(--muted2)", "marginTop": "4px"}),
            ]),
        ])

    def _period_list(title, periods, color):
        return html.Div(className="panel", style={"margin": "0"}, children=[
            html.Div(className="panel-header", children=[
                html.Span(title, className="panel-title"),
                html.Span(str(len(periods)), className=f"tag tag-{color}", style={"marginLeft": "auto"}),
            ]),
            html.Div(
                ", ".join(periods) if periods else "Aucune",
                style={"padding": "12px", "fontSize": "11px", "fontFamily": "var(--font-mono)",
                       "color": "var(--muted)", "lineHeight": "1.8"},
            ),
        ])

    return html.Div([
        html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "14px"},
                 children=[_card(ra), _card(rb)]),
        _period_list(f"Anomalies uniquement dans Run #{ra['id'] if ra else '?'}", cmp["only_a"], "red"),
        html.Div(style={"height": "10px"}),
        _period_list(f"Anomalies uniquement dans Run #{rb['id'] if rb else '?'}", cmp["only_b"], "red"),
        html.Div(style={"height": "10px"}),
        _period_list("Anomalies communes aux deux runs", cmp["common"], "grey"),
    ])


def render_historique(runs: list[dict] | None = None) -> html.Div:
    from database import models as _db
    runs = runs if runs is not None else _db.get_run_history(limit=50)

    return html.Div(children=[
        html.Div(className="topbar", children=[
            html.Span("Historique", className="tb-section"),
            html.Span("›", className="tb-sep"),
            html.Span("Runs et revue des anomalies", className="tb-page"),
            html.Span(f"{len(runs)} run{'s' if len(runs) > 1 else ''}", className="tb-context"),
        ]),
        html.Div(className="content", children=[
            html.Div(style={"display": "grid", "gridTemplateColumns": "320px 1fr",
                            "gap": "16px", "alignContent": "start", "height": "100%"}, children=[
                html.Div(className="panel", style={"margin": "0", "maxHeight": "100%", "overflowY": "auto"},
                         children=[
                    html.Div(className="panel-header", children=[
                        svg(ICO_CLOCK, size=12, stroke="var(--gold)"),
                        html.Span("Runs récents", className="panel-title"),
                    ]),
                    html.Div(id="hist-run-list", children=[_run_row(r, False, False) for r in runs]),
                ]),
                html.Div(className="panel", style={"margin": "0"}, children=[
                    html.Div(className="panel-header", children=[
                        svg(ICO_GRID, size=12, stroke="var(--gold)"),
                        html.Span("Détail", className="panel-title"),
                    ]),
                    html.Div(id="hist-right-panel",
                             children=html.Div(
                                 "Sélectionnez un run (« Voir ») pour afficher ses anomalies, "
                                 "ou deux runs (« Comparer ») pour les comparer.",
                                 style={"padding": "24px", "color": "var(--muted2)", "fontSize": "12px"},
                             )),
                ]),
            ]),
        ]),
        dcc.Store(id="store-hist-runs-cache", data=runs),
    ])
