"""Page Modèles  calibration LOF* et entraînement BiVAT."""
import os
from dash import html, dcc
from config import PAYS, PAYS_LABELS, VOLETS
from layout.icons import svg, ICO_PLAY, ICO_REFRESH, ICO_TERM, ICO_WARN


_INPUT_STYLE = {
    "width": "100%", "padding": "6px 10px",
    "background": "var(--surf2)", "border": "1px solid var(--border2)",
    "borderRadius": "3px", "color": "var(--text)",
    "fontFamily": "inherit", "fontSize": "12px",
}


def _num_input(label: str, id_: str, value, step=1, mini=None, full_width: bool = False):
    return html.Div(style={"gridColumn": "span 2" if full_width else None}, children=[
        html.Div(label, style={"fontSize": "10px", "color": "var(--muted2)", "marginBottom": "3px"}),
        dcc.Input(id=id_, type="number", value=value, step=step,
                  **({"min": mini} if mini is not None else {}),
                  style=_INPUT_STYLE),
    ])


def _text_input(label: str, id_: str, value: str):
    return html.Div([
        html.Div(label, style={"fontSize": "10px", "color": "var(--muted2)", "marginBottom": "3px"}),
        dcc.Input(id=id_, type="text", value=value, placeholder="AAAA-MM-JJ", style=_INPUT_STYLE),
    ])


def render_modeles(pays: str = "cameroun", volet: str = "Actif") -> html.Div:
    import torch
    from utils.hardware import DEVICE
    if DEVICE.type == "cuda":
        device_str = f"CUDA · {torch.cuda.get_device_name(0)}"
    elif DEVICE.type == "mps":
        device_str = "MPS · Apple Silicon"
    else:
        device_str = f"CPU · {os.cpu_count()} cœurs · PyTorch {torch.__version__}"

    return html.Div(children=[
        html.Div(className="topbar", children=[
            html.Span("Modèles", className="tb-section"),
            html.Span("›", className="tb-sep"),
            html.Span("Réentraînement & calibration", className="tb-page"),
            html.Span(f"Device : {device_str}", className="tb-context"),
        ]),
        html.Div(
            className="content",
            style={"padding": "10px 14px 0", "display": "flex", "gap": "12px", "alignItems": "flex-end"},
            children=[
                html.Div(style={"flex": "0 0 180px"}, children=[
                    html.Div("Pays", className="label"),
                    dcc.Dropdown(id="dd-modeles-pays",
                                 options=[{"label": PAYS_LABELS[p], "value": p} for p in PAYS],
                                 value=pays, clearable=False, className="select"),
                ]),
                html.Div(style={"flex": "0 0 140px"}, children=[
                    html.Div("Volet", className="label"),
                    dcc.Dropdown(id="dd-modeles-volet",
                                 options=[{"label": v, "value": v} for v in VOLETS],
                                 value=volet, clearable=False, className="select"),
                ]),
                html.Div("Les recalibrages/ré-entraînements ci-dessous s'appliquent au "
                         "pays/volet sélectionné ici (indépendant de la page Analyse).",
                         style={"fontSize": "10px", "color": "var(--muted2)", "paddingBottom": "6px"}),
            ],
        ),
        html.Div(className="content",
                 style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": "14px", "alignContent": "start"},
                 children=[
                     # ── LOF* card ─────────────────────────────────────────────
                     html.Div(className="panel", style={"margin": "0"}, children=[
                         html.Div(className="panel-header", children=[
                             html.Span("LOF*", className="tag tag-gold", style={"fontSize": "9px"}),
                             html.Span("Local Outlier Factor", className="panel-title", style={"marginLeft": "6px"}),
                             html.Span(id="lof-status-badge", className="tag tag-grey",
                                       style={"marginLeft": "auto", "fontSize": "9px"},
                                       children="Non calibré"),
                         ]),
                         html.Div(className="panel-body",
                                  style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                                  children=[
                                      html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "8px"},
                                               children=[
                                                   html.Div(className="stat-card", style={"padding": "10px"}, children=[html.Div("k ∈ [10,30]", className="stat-num", style={"fontSize": "16px"}), html.Div("MinPts range", className="stat-label")]),
                                                   html.Div(className="stat-card", style={"padding": "10px"}, children=[html.Div("", className="stat-num", style={"fontSize": "18px"}, id="lof-tau-display"), html.Div("Seuil τ moyen", className="stat-label")]),
                                                   html.Div(className="stat-card", style={"padding": "10px"}, children=[html.Div("195", className="stat-num", style={"fontSize": "18px", "color": "var(--green)"}), html.Div("Obs. traitées", className="stat-label")]),
                                               ]),
                                      html.Div([
                                          html.Div("Recalibrer le seuil τ", className="label"),
                                          dcc.Dropdown(id="dd-lof-tau",
                                                       options=[{"label": l, "value": v} for l, v in [
                                                           ("Q3 + 1,5 × IQR (défaut)", "iqr15"),
                                                           ("Q3 + 2,0 × IQR (conservateur)", "iqr20"),
                                                           ("Percentile 95%", "p95"),
                                                       ]],
                                                       value="iqr15", clearable=False, className="select"),
                                      ]),
                                      html.Button(
                                          id="btn-run-lof-all",
                                          className="btn btn-gold",
                                          style={"width": "100%", "justifyContent": "center"},
                                          children=[svg(ICO_REFRESH, size=12), " Recalibrer LOF*"],
                                          n_clicks=0,
                                      ),
                                      html.Div(id="lof-calibration-status",
                                               style={"padding": "10px", "background": "var(--surf2)",
                                                      "borderRadius": "4px", "border": "1px solid var(--border)"},
                                               children=[
                                                   html.Div("Statut calibration",
                                                            style={"fontSize": "10px", "color": "var(--muted)", "marginBottom": "4px"}),
                                                   html.Div("Aucune calibration effectuée dans cette session.",
                                                            style={"fontSize": "11px", "color": "var(--muted)"}),
                                               ]),
                                  ]),
                     ]),
                     # ── BiVAT card ────────────────────────────────────────────
                     html.Div(className="panel", style={"margin": "0"}, children=[
                         html.Div(className="panel-header", children=[
                             html.Span("BiVAT", className="tag tag-gold", style={"fontSize": "9px"}),
                             html.Span("Bi-Transformer VAE Temporal", className="panel-title", style={"marginLeft": "6px"}),
                             html.Span([svg(ICO_WARN, size=9, stroke="currentColor"), " Non entraîné"],
                                       className="tag tag-red",
                                       style={"marginLeft": "auto", "fontSize": "9px", "gap": "3px"}),
                         ]),
                         html.Div(className="panel-body",
                                  style={"display": "flex", "flexDirection": "column", "gap": "10px"},
                                  children=[
                                      html.Div("ENTRAÎNEMENT", className="label"),
                                      html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "8px"},
                                               children=[
                                                   _num_input("Époques",      "inp-bivat-epochs", 200, step=10, mini=1),
                                                   _num_input("Fenêtre (mois)", "inp-bivat-window", 12, step=1, mini=4),
                                                   _num_input("Learning rate", "inp-bivat-lr", 0.001, step=0.0001),
                                               ]),
                                      html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                                               children=[
                                                   _num_input("β (poids KL)", "inp-bivat-beta", 0.5, step=0.1),
                                                   _num_input("λ_ad (association discrepancy)", "inp-bivat-lad", 0.1, step=0.01),
                                               ]),
                                      html.Hr(style={"border": "none", "borderTop": "1px solid var(--border)", "margin": "2px 0"}),
                                      html.Div("CALIBRATION (Prédiction Conforme)", className="label"),
                                      html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                                               children=[
                                                   _text_input("Fin d'entraînement (train_end)", "inp-bivat-train-end", "2019-12-31"),
                                                   _text_input("Début de test (test_start)",      "inp-bivat-test-start", "2020-03-01"),
                                               ]),
                                      _num_input("Repli calibration (% du train, si calendrier insuffisant)",
                                                 "inp-bivat-cal-split", 0.20, step=0.05, full_width=True),
                                      html.Div(
                                          id="bivat-train-hint",
                                          style={"padding": "10px", "background": "rgba(212,160,32,.06)",
                                                 "borderRadius": "4px", "border": "1px solid var(--gold-md)",
                                                 "fontSize": "11px", "color": "var(--muted)"},
                                          children="Entraîner d'abord LOF* (page Analyse) pour le pays/volet "
                                                   "sélectionné ci-dessus, puis lancer BiVAT ici.",
                                      ),
                                      html.Button(
                                          id="btn-train-bivat",
                                          className="btn btn-solid-gold",
                                          style={"width": "100%", "justifyContent": "center"},
                                          n_clicks=0,
                                          children=[svg(ICO_PLAY, size=12), " Entraîner / recalibrer BiVAT"],
                                      ),
                                      dcc.ConfirmDialog(
                                          id="confirm-bivat-retrain",
                                          message="",
                                      ),
                                  ]),
                     ]),
                     # ── Journal d'entraînement ─────────────────────────────────
                     html.Div(className="panel", style={"margin": "0", "gridColumn": "span 2"}, children=[
                         html.Div(className="panel-header", children=[
                             svg(ICO_TERM, size=12, stroke="var(--gold)"),
                             html.Span("Journal d'entraînement", className="panel-title"),
                             html.Span(id="training-log-label", className="panel-sub",
                                       style={"fontFamily": "Roboto Mono, monospace", "fontSize": "10px"},
                                       children="En attente"),
                         ]),
                         html.Div(
                             id="training-log",
                             style={"padding": "10px 14px", "fontFamily": "Roboto Mono, monospace",
                                    "fontSize": "10px", "color": "var(--muted)", "lineHeight": "1.8",
                                    "background": "var(--surf2)", "maxHeight": "180px", "overflowY": "auto"},
                             children=[html.Div("En attente du lancement…",
                                                style={"color": "var(--muted2)"})],
                         ),
                     ]),
                 ]),
    ])
