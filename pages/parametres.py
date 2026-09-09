"""Page Paramètres style VS Code : nav catégories à gauche, contenu à droite."""
from dash import html, dcc
from layout.icons import (
    svg,
    ICO_SETTINGS, ICO_PALETTE, ICO_FOLDER, ICO_SHIELD, ICO_INFO,
    ICO_SUN, ICO_MOON, ICO_SEARCH, ICO_KEY, ICO_CHECK,
)
from config import PAYS, PAYS_LABELS, VOLETS


def _section_label(text: str):
    return html.Div(text, className="settings-section-label")


def _row(label: str, description: str, control):
    return html.Div(className="settings-row", children=[
        html.Div(className="settings-row-info", children=[
            html.Div(label, className="settings-row-label"),
            html.Div(description, className="settings-row-desc"),
        ]),
        html.Div(className="settings-row-control", children=[control]),
    ])


def _group(title: str, rows: list):
    return html.Div(className="settings-group", children=[
        html.Div(title, className="settings-group-title"),
        *rows,
    ])


def render_parametres(theme: str = "dark") -> html.Div:
    categories = [
        ("general",    ICO_SETTINGS, "Général"),
        ("apparence",  ICO_PALETTE,  "Apparence"),
        ("donnees",    ICO_FOLDER,   "Données"),
        ("securite",   ICO_SHIELD,   "Sécurité"),
        ("a-propos",   ICO_INFO,     "À propos"),
    ]

    nav_items = []
    for cat_id, ico, label in categories:
        nav_items.append(
            html.Div(
                id=f"settings-nav-{cat_id}",
                n_clicks=0,
                role="button", tabIndex=0,
                **({"aria-current": "true"} if cat_id == "general" else {}),
                className="settings-item active" if cat_id == "general" else "settings-item",
                children=[
                    html.Div(className="settings-item-icon", children=[svg(ico, size=14)]),
                    html.Div(label, className="settings-item-label"),
                ],
            )
        )

    # ── Contenu de chaque section ─────────────────────────────────────────────
    general_content = html.Div(id="settings-content-general", className="settings-content-section", children=[
        _section_label("Général"),
        _group("Interface", [
            _row("Langue", "Langue d'affichage de l'interface.",
                 dcc.Dropdown(
                     id="cfg-langue", value="fr",
                     options=[{"label": "Français", "value": "fr"},
                               {"label": "English",  "value": "en"}],
                     clearable=False, className="cfg-select",
                 )),
            _row("Pays par défaut", "Pays présélectionné à l'ouverture.",
                 dcc.Dropdown(
                     id="cfg-pays-defaut", value="cameroun",
                     options=[{"label": PAYS_LABELS[p], "value": p} for p in PAYS],
                     clearable=False, className="cfg-select",
                 )),
            _row("Volet par défaut", "Volet présélectionné à l'ouverture.",
                 dcc.Dropdown(
                     id="cfg-volet-defaut", value="Actif",
                     options=[{"label": v, "value": v} for v in VOLETS],
                     clearable=False, className="cfg-select",
                 )),
        ]),
        _group("Pipeline", [
            _row("Intervalle de polling", "Fréquence d'actualisation des résultats (ms).",
                 dcc.Dropdown(
                     id="cfg-poll-interval", value="2000",
                     options=[{"label": "500 ms",  "value": "500"},
                               {"label": "1 000 ms", "value": "1000"},
                               {"label": "2 000 ms", "value": "2000"},
                               {"label": "5 000 ms", "value": "5000"}],
                     clearable=False, className="cfg-select",
                 )),
        ]),
    ])

    apparence_content = html.Div(id="settings-content-apparence", className="settings-content-section",
                                  style={"display": "none"}, children=[
        _section_label("Apparence"),
        _group("Thème", [
            _row("Thème actif", "Bascule entre le thème clair et sombre.",
                 html.Div(className="theme-toggle-row", children=[
                     html.Div([svg(ICO_MOON, size=14), " Sombre"],
                              className="theme-pill", id="cfg-theme-dark",  n_clicks=0,
                              role="button", tabIndex=0),
                     html.Div([svg(ICO_SUN,  size=14), " Clair"],
                              className="theme-pill", id="cfg-theme-light", n_clicks=0,
                              role="button", tabIndex=0),
                 ])),
        ]),
        _group("Typographie", [
            _row("Police de l'interface", "Famille de police principale (redémarrage requis).",
                 dcc.Dropdown(
                     id="cfg-font", value="roboto",
                     options=[{"label": "Roboto (défaut)", "value": "roboto"},
                               {"label": "Inter",           "value": "inter"},
                               {"label": "System UI",       "value": "system"}],
                     clearable=False, className="cfg-select",
                 )),
            _row("Taille de base", "Taille de police en pixels.",
                 dcc.Dropdown(
                     id="cfg-font-size", value="13",
                     options=[{"label": f"{s} px", "value": str(s)} for s in [11, 12, 13, 14, 15]],
                     clearable=False, className="cfg-select",
                 )),
        ]),
    ])

    donnees_content = html.Div(id="settings-content-donnees", className="settings-content-section",
                                style={"display": "none"}, children=[
        _section_label("Données"),
        _group("Répertoire", [
            _row("Dossier de données", "Chemin vers le répertoire contenant les fichiers CSV.",
                 html.Div(className="cfg-path-row", children=[
                     html.Div("data/", id="cfg-data-path-display", className="cfg-path-display"),
                     html.Button([svg(ICO_FOLDER, size=12), " Parcourir"],
                                 id="cfg-browse-data", n_clicks=0, className="btn btn-ghost"),
                 ])),
        ]),
        _group("Format", [
            _row("Séparateur CSV", "Caractère séparateur dans les fichiers d'entrée.",
                 dcc.Dropdown(
                     id="cfg-csv-sep", value=";",
                     options=[{"label": "Point-virgule (;)", "value": ";"},
                               {"label": "Virgule (,)",      "value": ","},
                               {"label": "Tabulation (\\t)",  "value": "\\t"}],
                     clearable=False, className="cfg-select",
                 )),
            _row("Encodage", "Encodage des fichiers CSV.",
                 dcc.Dropdown(
                     id="cfg-encoding", value="utf-8-sig",
                     options=[{"label": "UTF-8 BOM",  "value": "utf-8-sig"},
                               {"label": "UTF-8",      "value": "utf-8"},
                               {"label": "Latin-1",    "value": "latin-1"}],
                     clearable=False, className="cfg-select",
                 )),
        ]),
    ])

    securite_content = html.Div(id="settings-content-securite", className="settings-content-section",
                                 style={"display": "none"}, children=[
        _section_label("Sécurité"),
        _group("Authentification", [
            _row("Délai de session", "Durée d'inactivité avant déconnexion automatique.",
                 dcc.Dropdown(
                     id="cfg-session-timeout", value="60",
                     options=[{"label": "15 min", "value": "15"},
                               {"label": "30 min", "value": "30"},
                               {"label": "60 min", "value": "60"},
                               {"label": "Jamais", "value": "0"}],
                     clearable=False, className="cfg-select",
                 )),
        ]),
        _group("Changer le mot de passe", [
            html.Div(className="settings-row", children=[
                html.Div(className="settings-row-info", children=[
                    html.Div("Mot de passe actuel", className="settings-row-label"),
                ]),
                html.Div(className="settings-row-control", children=[
                    dcc.Input(id="cfg-pwd-current", type="password",
                              placeholder="••••••••", className="settings-input"),
                ]),
            ]),
            html.Div(className="settings-row", children=[
                html.Div(className="settings-row-info", children=[
                    html.Div("Nouveau mot de passe", className="settings-row-label"),
                ]),
                html.Div(className="settings-row-control", children=[
                    dcc.Input(id="cfg-pwd-new", type="password",
                              placeholder="••••••••", className="settings-input"),
                ]),
            ]),
            html.Div(className="settings-row", children=[
                html.Div(className="settings-row-info", children=[
                    html.Div("Confirmation", className="settings-row-label"),
                ]),
                html.Div(className="settings-row-control", children=[
                    dcc.Input(id="cfg-pwd-confirm", type="password",
                              placeholder="••••••••", className="settings-input"),
                ]),
            ]),
            html.Div(style={"padding": "8px 0 0", "display": "flex", "gap": "8px"}, children=[
                html.Button([svg(ICO_KEY, size=12), " Changer"],
                            id="cfg-btn-change-pwd", n_clicks=0, className="btn btn-gold"),
                html.Div(id="cfg-pwd-feedback", className="settings-feedback"),
            ]),
        ]),
    ])

    apropos_content = html.Div(id="settings-content-a-propos", className="settings-content-section",
                                style={"display": "none"}, children=[
        _section_label("À propos"),
        html.Div(className="about-card", children=[
            html.Img(src="/assets/logo_beac.jfif", style={"width": "56px", "height": "56px",
                                                            "objectFit": "contain", "marginBottom": "12px"}),
            html.Div("BEAC · Détection d'anomalies", className="about-title"),
            html.Div("Version 1.0.0", className="about-version"),
            html.Div("Banque des États de l'Afrique Centrale CEMAC",
                     className="about-sub"),
        ]),
        html.Div(className="settings-group", children=[
            html.Div("Informations techniques", className="settings-group-title"),
            *[html.Div(className="about-row", children=[
                html.Span(k, className="about-row-key"),
                html.Span(v, className="about-row-val"),
              ]) for k, v in [
                  ("Moteur de détection", "LOF* (Local Outlier Factor adapté)"),
                  ("Modèle avancé",       "BiVAT (Bi-Variational Autoencoder Transformer)"),
                  ("Framework UI",        "Dash / Plotly + PyQt6"),
                  ("Python",              "3.10+"),
                  ("Licence",             "Usage interne BEAC"),
              ]],
        ]),
    ])

    return html.Div(className="settings-layout", children=[
        # Barre de recherche en haut
        html.Div(className="settings-search-bar", children=[
            html.Div(className="settings-search-icon", children=[svg(ICO_SEARCH, size=14)]),
            dcc.Input(id="settings-search", type="text",
                      placeholder="Rechercher dans les paramètres…",
                      className="settings-search-input", debounce=False),
        ]),
        html.Div(className="settings-body", children=[
            # Sidebar catégories
            html.Div(className="settings-nav", children=nav_items),
            # Zone de contenu
            html.Div(className="settings-content", children=[
                general_content,
                apparence_content,
                donnees_content,
                securite_content,
                apropos_content,
            ]),
        ]),
    ])
