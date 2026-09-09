"""Page Compte profil utilisateur, session réelle depuis store-auth."""
from dash import html, dcc
from layout.icons import (
    svg, ICO_USER, ICO_KEY, ICO_LOGOUT, ICO_SHIELD, ICO_CLOCK, ICO_INFO,
)


def render_compte(auth: dict | None = None) -> html.Div:
    auth      = auth or {}
    username  = auth.get("user",     "—")
    role      = auth.get("role",     "Analyste")
    login_at  = auth.get("login_at", "—")
    initials  = "".join(p[0].upper() for p in username.split(".")[:2]) if username != "—" else "AN"

    role_label = {"admin": "Administrateur", "analyste": "Analyste"}.get(role, role.capitalize())

    return html.Div(className="account-page", children=[
        # En-tête
        html.Div(className="account-header", children=[
            html.Div(initials, className="account-avatar-lg"),
            html.Div(className="account-header-info", children=[
                html.Div(username,    className="account-username"),
                html.Div(role_label,  className="account-role"),
            ]),
        ]),

        # Informations de compte
        html.Div(className="settings-group", children=[
            html.Div("Informations du compte", className="settings-group-title"),
            _info_row(ICO_USER,  "Identifiant",       username),
            _info_row(ICO_SHIELD,"Rôle",               role_label),
            _info_row(ICO_CLOCK, "Connecté depuis",    login_at),
        ]),

        # Sécurité changement de mot de passe
        html.Div(className="settings-group", children=[
            html.Div("Sécurité", className="settings-group-title"),
            html.Div(className="settings-row", children=[
                html.Div(className="settings-row-info", children=[
                    html.Div("Nouveau mot de passe", className="settings-row-label"),
                    html.Div("Doit contenir au moins 8 caractères.",
                             className="settings-row-desc"),
                ]),
                html.Div(className="settings-row-control", children=[
                    dcc.Input(id="acct-pwd-new", type="password",
                              placeholder="••••••••", className="settings-input"),
                ]),
            ]),
            html.Div(className="settings-row", children=[
                html.Div(className="settings-row-info", children=[
                    html.Div("Confirmation", className="settings-row-label"),
                ]),
                html.Div(className="settings-row-control", children=[
                    dcc.Input(id="acct-pwd-confirm", type="password",
                              placeholder="••••••••", className="settings-input"),
                ]),
            ]),
            html.Div(style={"padding": "8px 0 0", "display": "flex",
                             "gap": "8px", "alignItems": "center"}, children=[
                html.Button([svg(ICO_KEY, size=12), " Changer le mot de passe"],
                            id="acct-btn-change-pwd", n_clicks=0, className="btn btn-gold"),
                html.Div(id="acct-pwd-feedback", className="settings-feedback"),
            ]),
        ]),

        # Session déconnexion
        html.Div(className="settings-group", children=[
            html.Div("Session", className="settings-group-title"),
            html.Div(className="settings-row", children=[
                html.Div(className="settings-row-info", children=[
                    html.Div("Déconnexion", className="settings-row-label"),
                    html.Div("Terminer la session et revenir à l'écran de connexion.",
                             className="settings-row-desc"),
                ]),
                html.Div(className="settings-row-control", children=[
                    html.Button([svg(ICO_LOGOUT, size=12), " Se déconnecter"],
                                id="btn-logout", n_clicks=0, className="btn btn-red"),
                ]),
            ]),
        ]),
    ])


def _info_row(ico_path: str, label: str, value: str):
    return html.Div(className="account-info-row", children=[
        html.Div(className="account-info-icon",
                 children=[svg(ico_path, size=14)]),
        html.Div(label,  className="account-info-label"),
        html.Div(value,  className="account-info-value"),
    ])
