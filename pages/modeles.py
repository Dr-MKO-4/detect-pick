"""Page Modèles  calibration LOF* et entraînement BiVAT."""
import os
from dash import html, dcc
from layout.icons import svg, ICO_PLAY, ICO_REFRESH, ICO_TERM, ICO_WARN


def render_modeles() -> html.Div:
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
                                  style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                                  children=[
                                      html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "8px"},
                                               children=[
                                                   html.Div(className="stat-card", style={"padding": "10px"}, children=[html.Div("200", className="stat-num", style={"fontSize": "18px"}), html.Div("Époques", className="stat-label")]),
                                                   html.Div(className="stat-card", style={"padding": "10px"}, children=[html.Div("64",  className="stat-num", style={"fontSize": "18px"}), html.Div("d_model", className="stat-label")]),
                                                   html.Div(className="stat-card", style={"padding": "10px"}, children=[html.Div("16",  className="stat-num", style={"fontSize": "18px"}), html.Div("Dim. latente", className="stat-label")]),
                                               ]),
                                      html.Div([
                                          html.Div("Fenêtre d'entraînement", className="label"),
                                          dcc.Dropdown(id="dd-bivat-window",
                                                       options=[{"label": l, "value": v} for l, v in [
                                                           ("Déc 2001 → Déc 2019 (pré-test)", "pretrain"),
                                                           ("Déc 2001 → Mar 2026 (complète)", "full"),
                                                       ]],
                                                       value="pretrain", clearable=False, className="select"),
                                      ]),
                                      html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
                                               children=[
                                                   html.Div([
                                                       html.Div("β (poids KL)", style={"fontSize": "10px", "color": "var(--muted2)", "marginBottom": "3px"}),
                                                       dcc.Input(id="inp-bivat-beta", type="number", value=0.5, step=0.1,
                                                                 style={"width": "100%", "padding": "6px 10px",
                                                                        "background": "var(--surf2)",
                                                                        "border": "1px solid var(--border2)",
                                                                        "borderRadius": "3px", "color": "var(--text)",
                                                                        "fontFamily": "inherit", "fontSize": "12px"}),
                                                   ]),
                                                   html.Div([
                                                       html.Div("λ_ad (discrepancy)", style={"fontSize": "10px", "color": "var(--muted2)", "marginBottom": "3px"}),
                                                       dcc.Input(id="inp-bivat-lad", type="number", value=0.1, step=0.01,
                                                                 style={"width": "100%", "padding": "6px 10px",
                                                                        "background": "var(--surf2)",
                                                                        "border": "1px solid var(--border2)",
                                                                        "borderRadius": "3px", "color": "var(--text)",
                                                                        "fontFamily": "inherit", "fontSize": "12px"}),
                                                   ]),
                                               ]),
                                      html.Div(
                                          style={"padding": "10px", "background": "rgba(212,160,32,.06)",
                                                 "borderRadius": "4px", "border": "1px solid var(--gold-md)",
                                                 "fontSize": "11px", "color": "var(--muted)"},
                                          children="Entraîner d'abord LOF* (page Analyse) puis revenir ici pour lancer BiVAT.",
                                      ),
                                      html.Button(
                                          id="btn-train-bivat",
                                          className="btn btn-solid-gold",
                                          style={"width": "100%", "justifyContent": "center"},
                                          disabled=True,
                                          n_clicks=0,
                                          children=[svg(ICO_PLAY, size=12), " Entraîner BiVAT (à venir)"],
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
