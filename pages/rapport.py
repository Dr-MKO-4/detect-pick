"""Page Rapport génération et historique réel depuis SQLite."""
from dash import html, dcc
from config import PAYS, PAYS_LABELS
from layout.icons import svg, ICO_GRID, ICO_CLOCK, ICO_PULSE, ICO_DL, ICO_CHECK


def _history_rows(reports: list[dict]) -> list:
    if not reports:
        return [html.Div("Aucun rapport généré.",
                         style={"padding": "14px", "fontSize": "11px",
                                "color": "var(--muted)"})]
    rows = []
    for r in reports:
        fname = (r.get("file_path") or "").split("/")[-1].split("\\")[-1] or r.get("report_type", "—")
        user  = r.get("username") or "—"
        date  = (r.get("created_at") or "")[:16]
        rows.append(html.Div(
            style={"padding": "8px 14px", "borderBottom": "1px solid var(--border)",
                   "display": "flex", "alignItems": "center", "gap": "8px"},
            children=[
                html.Div(svg(ICO_CHECK, size=10, stroke="var(--green)"),
                         style={"flexShrink": "0"}),
                html.Span(fname,
                          style={"flex": "1", "fontSize": "11px", "color": "var(--text)",
                                 "overflow": "hidden", "textOverflow": "ellipsis",
                                 "whiteSpace": "nowrap"}),
                html.Span(user,
                          style={"fontSize": "9px", "color": "var(--muted)",
                                 "fontFamily": "var(--font-mono)"}),
                html.Span(date,
                          style={"fontSize": "9px", "color": "var(--muted2)",
                                 "fontFamily": "var(--font-mono)", "flexShrink": "0"}),
            ],
        ))
    return rows


def render_rapport(reports: list[dict] | None = None) -> html.Div:
    reports = reports or []
    n_reports = len(reports)

    last_gen = (
        f"Dernier : {reports[0]['created_at'][:16]}" if reports
        else "Aucun rapport généré dans cette session"
    )

    return html.Div(children=[
        html.Div(className="topbar", children=[
            html.Span("Rapport", className="tb-section"),
            html.Span("›", className="tb-sep"),
            html.Span("Génération de rapport", className="tb-page"),
            html.Span(id="rapport-last-gen", className="tb-context", children=last_gen),
        ]),
        html.Div(className="content",
                 style={"display": "grid", "gridTemplateColumns": "300px 1fr",
                        "gap": "16px", "alignContent": "start"},
                 children=[
                     # Colonne config
                     html.Div(style={"display": "flex", "flexDirection": "column", "gap": "14px"},
                              children=[
                         html.Div(className="panel", style={"margin": "0"}, children=[
                             html.Div(className="panel-header", children=[
                                 svg(ICO_GRID, size=12, stroke="var(--gold)"),
                                 html.Span("Configuration", className="panel-title"),
                             ]),
                             html.Div(className="panel-body",
                                      style={"display": "flex", "flexDirection": "column",
                                             "gap": "12px"},
                                      children=[
                                 html.Div([
                                     html.Div("Type de rapport", className="label"),
                                     dcc.Dropdown(
                                         id="dd-rapport-type",
                                         options=[{"label": l, "value": l} for l in [
                                             "Rapport complet CEMAC", "Rapport par pays",
                                             "Synthèse anomalies", "Rapport technique LOF",
                                             "Rapport technique BiVAT", "Rapport comparatif",
                                         ]],
                                         value="Rapport complet CEMAC",
                                         clearable=False, className="select",
                                     ),
                                 ]),
                                 html.Div([
                                     html.Div("Périmètre pays", className="label"),
                                     dcc.Dropdown(
                                         id="dd-rapport-pays",
                                         options=[{"label": l, "value": v} for l, v in
                                                  [("Tous (CEMAC)", "all")] +
                                                  [(PAYS_LABELS[p], p) for p in PAYS]],
                                         value="all", clearable=False, className="select",
                                     ),
                                 ]),
                                 html.Div([
                                     html.Div("Modèle(s)", className="label"),
                                     dcc.Dropdown(
                                         id="dd-rapport-modele",
                                         options=[
                                             {"label": "LOF* + BiVAT (comparaison)", "value": "both"},
                                             {"label": "LOF* seulement",             "value": "lof"},
                                             {"label": "BiVAT seulement",            "value": "bivat"},
                                         ],
                                         value="lof", clearable=False, className="select",
                                     ),
                                 ]),
                                 html.Div([
                                     html.Div("Format d'export", className="label"),
                                     dcc.Checklist(
                                         id="chk-rapport-format",
                                         options=[
                                             {"label": " HTML",  "value": "html"},
                                             {"label": " PDF",   "value": "pdf"},
                                             {"label": " Excel", "value": "xlsx"},
                                         ],
                                         value=["html"],
                                         style={"fontSize": "12px", "display": "flex", "gap": "12px"},
                                     ),
                                 ]),
                             ]),
                         ]),
                         html.Div(id="rapport-status",
                                  style={"fontSize": "11px", "color": "var(--muted)",
                                         "minHeight": "16px"}),
                         html.Button(
                             id="btn-generate-rapport",
                             className="btn btn-solid-gold",
                             style={"width": "100%", "padding": "10px",
                                    "justifyContent": "center", "fontSize": "13px"},
                             children=[svg(ICO_DL, size=14, sw=2.5), " Générer le rapport"],
                             n_clicks=0,
                         ),
                         dcc.Download(id="dl-rapport"),
                         html.Span(id="rapport-btn-dummy", style={"display": "none"}),
                     ]),

                     # Colonne historique + aperçu
                     html.Div(style={"display": "flex", "flexDirection": "column", "gap": "14px"},
                              children=[
                         html.Div(className="panel", style={"margin": "0"}, children=[
                             html.Div(className="panel-header", children=[
                                 svg(ICO_CLOCK, size=12, stroke="var(--gold)"),
                                 html.Span("Historique des rapports", className="panel-title"),
                                 html.Span(
                                     f"{n_reports} rapport{'s' if n_reports != 1 else ''}",
                                     className="panel-sub",
                                     id="rapport-history-count",
                                 ),
                             ]),
                             html.Div(id="rapport-history-list",
                                      children=_history_rows(reports)),
                         ]),
                         html.Div(className="panel", style={"margin": "0"}, children=[
                             html.Div(className="panel-header", children=[
                                 svg(ICO_PULSE, size=12, stroke="var(--gold)"),
                                 html.Span("Aperçu du rapport", className="panel-title"),
                             ]),
                             html.Div(
                                 style={"background": "var(--surf2)", "padding": "16px",
                                        "display": "flex", "justifyContent": "center",
                                        "minHeight": "240px"},
                                 children=[
                                     html.Div(
                                         style={"width": "360px", "background": "#fff",
                                                "padding": "24px",
                                                "boxShadow": "0 4px 16px rgba(0,0,0,.12)",
                                                "fontFamily": "sans-serif"},
                                         children=[
                                             html.Div(
                                                 style={"display": "flex", "alignItems": "center",
                                                        "gap": "10px", "marginBottom": "16px",
                                                        "paddingBottom": "10px",
                                                        "borderBottom": "2px solid #9e6f1a"},
                                                 children=[
                                                     html.Div("BEAC · CEMAC",
                                                              style={"fontSize": "11px",
                                                                     "fontWeight": "700",
                                                                     "color": "#1a1a1a"}),
                                                 ],
                                             ),
                                             html.Div("Rapport d'anomalies monétaires",
                                                      style={"fontSize": "15px", "fontWeight": "700",
                                                             "color": "#1a1a1a", "marginBottom": "4px"}),
                                             html.Div("Modèles LOF* + BiVAT · 6 pays CEMAC",
                                                      style={"fontSize": "9px", "color": "#666",
                                                             "marginBottom": "14px"}),
                                             html.Div(style={"height": "2px",
                                                             "background": "#9e6f1a",
                                                             "marginBottom": "10px"}),
                                             html.Div(
                                                 id="rapport-preview-text",
                                                 children="Configurez les options puis cliquez sur « Générer ».",
                                                 style={"fontSize": "8px", "color": "#aaa",
                                                        "lineHeight": "1.6"},
                                             ),
                                         ],
                                     ),
                                 ],
                             ),
                         ]),
                     ]),
                 ]),
    ])
