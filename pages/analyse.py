"""Page Analyse  figures LOF*/BiVAT par phase."""
from datetime import datetime as _dt
from dash import html, dcc
from config import PAYS, PAYS_LABELS, VOLETS
from layout.icons import svg, ICO_PULSE, ICO_PLAY, ICO_STOP, ICO_CHECK, ICO_WARN

_fig_cache: dict = {}   # (fig_id, json_prefix) → go.Figure  évite pio.from_json répété

def _cached_fig(fig_id: str, fig_json: str):
    import plotly.io as pio
    key = (fig_id, fig_json[:128])
    if key not in _fig_cache:
        if len(_fig_cache) > 60:
            _fig_cache.pop(next(iter(_fig_cache)))
        _fig_cache[key] = pio.from_json(fig_json)
    return _fig_cache[key]

_LOF_PHASE_FIGS: dict[int, list[tuple]] = {
    1: [
        ("fig1",  "FIG. 1",  "Carte des données manquantes",                     "span 2"),
        ("fig2",  "FIG. 2",  "Heatmap corrélations résidus STL",                 ""),
        ("fig3",  "FIG. 3",  "Décomposition RobustSTL  4 panneaux",             ""),
        ("figA",  "FIG. A",  "Variance RPCA sur fenêtres glissantes 36 mois",    "span 2"),
    ],
    2: [
        ("fig4",  "FIG. 4",  "Scree plot RPCA / ACP",                            ""),
        ("fig5",  "FIG. 5",  "Biplot ACP (CP1 × CP2)",                           ""),
        ("fig6",  "FIG. 6",  "Projection UMAP",                                  ""),
        ("fig7",  "FIG. 7",  "Profils LOF(MinPts)  top-20 anomalies",           "span 2"),
        ("fig8",  "FIG. 8",  "Distribution empirique LOF* + seuils τ",           "span 2"),
    ],
    3: [
        ("fig9",  "FIG. 9",  "Boxplot LOF* par année",                           "span 2"),
        ("fig10", "FIG. 10", "Série temporelle LOF*(t)",                         "span 2"),
        ("fig11", "FIG. 11", "LOF* vs indicateur brut (interactif)",             "span 2"),
        ("fig12", "FIG. 12", "Bar chart indicateurs contributeurs",              ""),
        ("fig13", "FIG. 13", "Heatmap anomalies  indicateurs × mois",           ""),
        ("fig14", "FIG. 14", "Carte de chaleur inter-pays",                      "span 2"),
        ("figB",  "FIG. B",  "Concordance Actif–Passif",                         "span 2"),
    ],
    4: [],
}

_BIVAT_PHASE_FIGS: dict[int, list[tuple]] = {
    1: [],
    2: [("figE",     "FIG. E",     "Score BiVAT + intervalles CP  calibration",  "span 2")],
    3: [],
    4: [
        ("figD",     "FIG. D",     "Comparaison LOF vs BiVAT normalisés [0,1]",   "span 2"),
        ("figE_bis", "FIG. E bis", "Score BiVAT + CP rolling  fenêtre test",     "span 2"),
        ("figF",     "FIG. F",     "Attribution SHAP  top-5 anomalies BiVAT",    ""),
        ("figG",     "FIG. G",     "Concordance LOF / BiVAT  6 pays × 4 types",  ""),
    ],
}

_PHASE_LABELS = {
    1: "Phase 1  Prétraitement",
    2: "Phase 2  Calibration",
    3: "Phase 3  Résultats",
    4: "Phase 4  Comparaison inter-modèles",
}

_PHASE_BTN_LABELS = {
    1: "Prétraitement",
    2: "Calibration",
    3: "Résultats",
    4: "Inter-modèles",
}


def analyse_controls(running: bool = False, modele: str = "lof",
                     progress: int = 0, progress_label: str = "",
                     pays: str = "cameroun", volet: str = "Actif",
                     annee: str = "all") -> html.Div:
    return html.Div(className="analyse-controls-panel", children=[
        # Section: Sélection
        html.Div("SÉLECTION", className="label", style={"marginBottom": "8px"}),
        html.Div([
            html.Div("Pays", className="label"),
            dcc.Dropdown(
                id="dd-pays",
                options=[{"label": PAYS_LABELS[p], "value": p} for p in PAYS],
                value=pays, clearable=False, className="select",
            ),
        ]),
        html.Div([
            html.Div("Année", className="label"),
            dcc.Dropdown(
                id="dd-annee",
                options=[{"label": "Toutes", "value": "all"}] + [
                    {"label": str(y), "value": str(y)}
                    for y in range(_dt.now().year, 2009, -1)
                ],
                value=annee, clearable=False, className="select",
            ),
        ]),
        html.Div([
            html.Div("Volet", className="label"),
            dcc.Dropdown(
                id="dd-volet",
                options=[{"label": v, "value": v} for v in VOLETS],
                value=volet, clearable=False, className="select",
            ),
        ]),
        html.Hr(style={"border": "none", "borderTop": "1px solid var(--border)", "margin": "8px 0"}),
        # Section: Modèle
        html.Div("MODÈLE", className="label", style={"marginBottom": "8px"}),
        html.Div(className="seg-ctrl", children=[
            html.Button(
                "LOF*", id="btn-model-lof", n_clicks=0,
                className="seg-btn" + (" seg-btn-active" if modele == "lof" else ""),
            ),
            html.Button(
                "BiVAT", id="btn-model-bivat", n_clicks=0,
                className="seg-btn" + (" seg-btn-active" if modele == "bivat" else ""),
            ),
        ]),
        dcc.Dropdown(
            id="dd-modele",
            value=modele,
            options=[{"label": "LOF*", "value": "lof"}, {"label": "BiVAT", "value": "bivat"}],
            style={"display": "none"},
            clearable=False,
        ),
        html.Hr(style={"border": "none", "borderTop": "1px solid var(--border)", "margin": "8px 0"}),
        # Run / Stop buttons
        html.Button(
            [svg(ICO_PLAY, 12), " Exécuter"],
            id="btn-run-pipeline", n_clicks=0,
            className="btn btn-solid-gold",
            style={"width": "100%", "justifyContent": "center",
                   "display": "none" if running else "flex"},
        ),
        html.Button(
            [svg(ICO_STOP, 12), " Arrêter"],
            id="btn-stop-pipeline", n_clicks=0,
            className="btn btn-red",
            style={"width": "100%", "justifyContent": "center",
                   "display": "flex" if running else "none"},
        ),
        # Progress bar (visible only when running)
        html.Div(id="pg-running-panel", style={"display": "block" if running else "none"}, children=[
            # Step indicators
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "0", "marginBottom": "8px"},
                children=[
                    html.Div("Charge", id="pg-step-charge", className=f"vs-step {'vs-step-done' if progress >= 20 else 'vs-step-active' if progress >= 5 else ''}"),
                    html.Div(className="vs-step-line"),
                    html.Div("LOF", id="pg-step-lof", className=f"vs-step {'vs-step-done' if progress >= 65 else 'vs-step-active' if progress >= 20 else ''}"),
                    html.Div(className="vs-step-line"),
                    html.Div("Figures", id="pg-step-figs", className=f"vs-step {'vs-step-done' if progress >= 95 else 'vs-step-active' if progress >= 65 else ''}"),
                ],
            ),
            # Progress track
            html.Div(className="vs-progress-track", children=[
                html.Div(id="pg-fill", className="vs-progress-fill", style={"width": f"{progress}%"}),
            ]),
            # Labels
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "marginTop": "4px"},
                children=[
                    html.Span(
                        id="pg-label",
                        children=progress_label or "En cours…",
                        style={"fontSize": "10px", "color": "var(--muted)",
                               "fontFamily": "Roboto Mono,monospace",
                               "overflow": "hidden", "textOverflow": "ellipsis",
                               "whiteSpace": "nowrap", "maxWidth": "140px"},
                    ),
                    html.Span(
                        id="pg-pct",
                        children=f"{progress}%",
                        style={"fontSize": "10px", "color": "var(--gold)",
                               "fontFamily": "Roboto Mono,monospace", "flexShrink": "0"},
                    ),
                ],
            ),
        ]),
    ])


def render_analyse(pays: str = "cameroun", volet: str = "Actif",
                   modele: str = "lof", phase: int = 1,
                   running: bool = False, results=None,
                   bivat_results=None,
                   progress: int = 0, progress_label: str = "",
                   annee: str = "all") -> html.Div:
    is_bivat    = modele == "bivat"
    model_label = "BiVAT" if is_bivat else "LOF*"
    valid_phases = [2, 4] if is_bivat else [1, 2, 3]

    phase_btns = []
    for i in range(1, 5):
        disabled = i not in valid_phases
        phase_btns.append(html.Button(
            _PHASE_BTN_LABELS[i],
            className="btn btn-gold" if i == phase else "btn btn-ghost",
            id=f"btn-phase-{i}",
            disabled=disabled,
            style={"fontSize": "11px", "padding": "5px 10px",
                   "opacity": "0.3" if disabled else "1"},
        ))

    fig_specs = (_BIVAT_PHASE_FIGS if is_bivat else _LOF_PHASE_FIGS).get(phase, [])
    lof_figs   = (results or {}).get("figures_json") or {}
    bivat_figs = (bivat_results or {}).get("figures_json") or {}
    figs_json  = bivat_figs if is_bivat else lof_figs

    fig_panels = []
    for fig_id, label, title, col_span in fig_specs:
        style = {"margin": "0"}
        if col_span:
            style["gridColumn"] = col_span

        fig_json = figs_json.get(fig_id)
        if fig_json:
            try:
                fig_obj = _cached_fig(fig_id, fig_json)
                # uirevision stable par figure : sans ça, Plotly réinitialise le
                # zoom/pan et l'état des boutons du modebar (ex. hovermode,
                # légende repliée) à chaque re-rendu de la page (changement de
                # phase, de résultats, de statut du pipeline…) même si la figure
                # elle-même n'a pas changé.
                fig_obj.update_layout(uirevision=fig_id)
                body = dcc.Graph(
                    id=f"graph-{fig_id}",
                    figure=fig_obj,
                    config={"displayModeBar": True, "responsive": True,
                            "toImageButtonOptions": {"format": "png", "scale": 2}},
                    style={"minHeight": "340px"},
                )
                tag = html.Span([svg(ICO_CHECK, size=9, stroke="currentColor"), " Calculé"],
                                className="tag tag-green",
                                style={"marginLeft": "8px", "fontSize": "9px", "gap": "3px"})
            except Exception as exc:
                body = html.Div(f"Erreur rendu : {exc}",
                                style={"padding": "14px", "color": "var(--red)", "fontSize": "11px"})
                tag = html.Span([svg(ICO_WARN, size=9, stroke="currentColor"), " Erreur"],
                                className="tag tag-red",
                                style={"marginLeft": "8px", "fontSize": "9px", "gap": "3px"})
        else:
            wait_msg = "En cours…" if running else "Lancez l'analyse"
            body = html.Div(
                style={"height": "130px", "display": "flex", "alignItems": "center",
                       "justifyContent": "center", "color": "var(--muted2)", "fontSize": "11px"},
                children=wait_msg,
            )
            tag = html.Span(wait_msg, className="tag tag-grey",
                            style={"marginLeft": "8px", "fontSize": "9px"})

        fig_panels.append(html.Div(className="panel", style=style, children=[
            html.Div(className="panel-header", children=[
                html.Span(label, style={"fontSize": "9px", "color": "var(--gold)",
                                        "fontFamily": "Roboto Mono, monospace"}),
                html.Span(title, className="panel-title", style={"marginLeft": "4px"}),
                tag,
            ]),
            body,
        ]))

    if not fig_panels:
        fig_panels = [html.Div(
            style={"gridColumn": "span 2", "padding": "40px", "textAlign": "center",
                   "color": "var(--muted2)", "fontSize": "12px"},
            children="Cette phase n'a pas de figures pour le modèle sélectionné.",
        )]

    notice = None
    if is_bivat and not results:
        notice = html.Div(
            style={"padding": "12px 16px", "background": "rgba(212,160,32,.08)",
                   "border": "1px solid var(--gold-md)", "borderRadius": "5px",
                   "fontSize": "11px", "color": "var(--gold)", "marginBottom": "14px",
                   "gridColumn": "span 2"},
            children=[svg(ICO_WARN, size=11, stroke="var(--gold)"),
                     " Exécutez d'abord LOF* pour ce pays/volet avant de lancer BiVAT."],
        )

    topbar_div = html.Div(className="topbar", children=[
        html.Span("Analyse", className="tb-section"),
        html.Span("›", className="tb-sep"),
        html.Span(_PHASE_LABELS.get(phase, f"Phase {phase}"), className="tb-page"),
        html.Span([
            html.Span(className="running-dot"), " En cours",
        ], className="tag tag-red", style={"marginLeft": "8px", "gap": "5px"}) if running else None,
        html.Span(f"{PAYS_LABELS.get(pays, pays)} · {volet} · {model_label}",
                  className="tb-context"),
        html.Div(phase_btns, style={"display": "flex", "gap": "6px", "marginLeft": "8px"}),
    ])

    return html.Div(style={"display": "flex", "flexDirection": "column", "height": "100%"}, children=[
        topbar_div,
        html.Div(style={"display": "flex", "flex": "1", "minHeight": "0", "overflow": "hidden"}, children=[
            analyse_controls(running=running, modele=modele,
                             progress=progress, progress_label=progress_label,
                             pays=pays, volet=volet, annee=annee),
            html.Div(className="content", style={"flex": "1"}, children=[
                html.Div(
                    className="grid-figs",
                    children=([notice] if notice else []) + fig_panels,
                ),
            ]),
        ]),
    ])
