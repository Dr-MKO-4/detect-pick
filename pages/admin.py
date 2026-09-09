"""Page Administration gestion des utilisateurs et journal d'audit.
Accessible uniquement aux utilisateurs avec le rôle 'admin'.
"""
from dash import html, dcc
from database import models as _db


_ROLE_LABELS = {"admin": "Administrateur", "analyste": "Analyste"}


def _badge(text: str, color: str = "grey") -> html.Span:
    return html.Span(text, className=f"tag tag-{color}",
                     style={"fontSize": "10px", "padding": "2px 6px"})


def _users_tab(users: list[dict]) -> html.Div:
    rows = []
    for u in users:
        is_active = bool(u["is_active"])
        status_badge = _badge("Actif", "green") if is_active else _badge("Inactif", "red")
        role_badge   = _badge(_ROLE_LABELS.get(u["role"], u["role"]),
                              "gold" if u["role"] == "admin" else "grey")
        rows.append(html.Tr([
            html.Td(str(u["id"]),    style={"width": "40px", "color": "var(--muted)"}),
            html.Td(u["username"],   style={"fontFamily": "Roboto Mono, monospace", "fontSize": "12px"}),
            html.Td(role_badge),
            html.Td(u.get("last_login", "—") or "—",
                    style={"fontSize": "11px", "color": "var(--muted)"}),
            html.Td(status_badge),
            html.Td([
                html.Button(
                    "Désactiver" if is_active else "Réactiver",
                    id={"type": "btn-toggle-user", "index": u["id"]},
                    n_clicks=0,
                    className="btn btn-sm btn-red" if is_active else "btn btn-sm",
                    style={"fontSize": "10px", "padding": "3px 8px", "marginRight": "6px"},
                ),
                html.Button(
                    "Réinit. MDP",
                    id={"type": "btn-reset-pwd", "index": u["id"]},
                    n_clicks=0,
                    className="btn btn-sm",
                    style={"fontSize": "10px", "padding": "3px 8px"},
                ),
            ], style={"whiteSpace": "nowrap"}),
        ]))

    return html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("ID"),
                html.Th("Nom d'utilisateur"),
                html.Th("Rôle"),
                html.Th("Dernière connexion"),
                html.Th("Statut"),
                html.Th("Actions"),
            ], style={"fontSize": "10px", "textTransform": "uppercase",
                      "letterSpacing": ".08em", "color": "var(--muted2)"})),
            html.Tbody(rows, id="admin-users-tbody"),
        ], className="data-table", style={"width": "100%"}),
        html.Div(id="admin-user-msg", style={"marginTop": "8px", "fontSize": "11px", "color": "var(--green)"}),
        dcc.Store(id="store-admin-pending", data=None),
    ])


def _create_tab() -> html.Div:
    return html.Div([
        html.Div("Créer un compte utilisateur",
                 style={"fontSize": "12px", "fontWeight": "600",
                        "color": "var(--muted)", "marginBottom": "16px",
                        "textTransform": "uppercase", "letterSpacing": ".08em"}),
        html.Div([
            html.Div([
                html.Label("Nom d'utilisateur",
                           style={"fontSize": "10px", "color": "var(--muted2)",
                                  "textTransform": "uppercase", "letterSpacing": ".08em",
                                  "marginBottom": "4px", "display": "block"}),
                dcc.Input(id="inp-new-username", type="text", value="",
                          placeholder="prenom.nom",
                          className="form-input",
                          style={"width": "100%", "marginBottom": "12px"}),
                html.Div("Identifiant de connexion. Utiliser le format prenom.nom.",
                         style={"fontSize": "10px", "color": "var(--muted2)", "marginBottom": "12px"}),

                html.Label("Mot de passe temporaire",
                           style={"fontSize": "10px", "color": "var(--muted2)",
                                  "textTransform": "uppercase", "letterSpacing": ".08em",
                                  "marginBottom": "4px", "display": "block"}),
                dcc.Input(id="inp-new-password", type="password", value="",
                          placeholder="Minimum 6 caractères",
                          className="form-input",
                          style={"width": "100%", "marginBottom": "12px"}),
                html.Div("Le mot de passe doit être transmis manuellement à l'utilisateur.",
                         style={"fontSize": "10px", "color": "var(--muted2)", "marginBottom": "12px"}),

                html.Label("Rôle",
                           style={"fontSize": "10px", "color": "var(--muted2)",
                                  "textTransform": "uppercase", "letterSpacing": ".08em",
                                  "marginBottom": "4px", "display": "block"}),
                dcc.Dropdown(id="dd-new-role",
                             options=[
                                 {"label": "Analyste", "value": "analyste"},
                                 {"label": "Administrateur", "value": "admin"},
                             ],
                             value="analyste",
                             clearable=False,
                             className="dd-input",
                             style={"marginBottom": "4px"}),
                html.Div("Les analystes n'ont pas accès à cette page d'administration.",
                         style={"fontSize": "10px", "color": "var(--muted2)", "marginBottom": "20px"}),

                html.Button("Créer l'utilisateur",
                            id="btn-create-user", n_clicks=0,
                            className="btn btn-gold",
                            style={"minWidth": "180px"}),
                html.Div(id="create-user-msg",
                         style={"marginTop": "10px", "fontSize": "11px"}),
            ], style={"maxWidth": "400px"}),
        ]),
    ])


def _audit_tab(audit: list[dict]) -> html.Div:
    rows = []
    for ev in audit[:50]:
        details = ""
        try:
            d = ev.get("details_json")
            if d:
                details = ", ".join(f"{k}: {v}" for k, v in __import__("json").loads(d).items())
        except Exception:
            pass
        rows.append(html.Tr([
            html.Td(ev.get("event_time", "")[:16],
                    style={"fontFamily": "Roboto Mono, monospace", "fontSize": "10px",
                           "color": "var(--muted)", "whiteSpace": "nowrap"}),
            html.Td(ev.get("username") or "système",
                    style={"fontSize": "11px"}),
            html.Td(ev.get("action", ""),
                    style={"fontSize": "11px", "color": "var(--gold)"}),
            html.Td(details,
                    style={"fontSize": "10px", "color": "var(--muted2)",
                           "maxWidth": "320px", "overflow": "hidden",
                           "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
        ]))

    return html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("Horodatage"),
                html.Th("Utilisateur"),
                html.Th("Action"),
                html.Th("Détails"),
            ], style={"fontSize": "10px", "textTransform": "uppercase",
                      "letterSpacing": ".08em", "color": "var(--muted2)"})),
            html.Tbody(rows),
        ], className="data-table", style={"width": "100%"}),
        html.Div(f"{len(audit)} événements au total",
                 style={"marginTop": "8px", "fontSize": "10px", "color": "var(--muted2)"}),
    ])


def _system_tab() -> html.Div:
    threshold = _db.get_app_setting("alert_threshold_anomalies", "5")
    return html.Div([
        html.Div("Alertes automatiques",
                 style={"fontSize": "12px", "fontWeight": "600",
                        "color": "var(--muted)", "marginBottom": "16px",
                        "textTransform": "uppercase", "letterSpacing": ".08em"}),
        html.Div(className="settings-row", style={"maxWidth": "500px", "borderBottom": "none"}, children=[
            html.Div(className="settings-row-info", children=[
                html.Div("Seuil de notification", className="settings-row-label"),
                html.Div("Nombre d'anomalies détectées sur un run à partir duquel "
                         "une notification est diffusée à tous les utilisateurs.",
                         className="settings-row-desc"),
            ]),
            html.Div(className="settings-row-control", children=[
                dcc.Input(id="inp-alert-threshold", type="number", min=1, step=1,
                         value=int(threshold) if threshold else 5,
                         className="form-input", style={"width": "100px"}),
            ]),
        ]),
        html.Button("Enregistrer", id="btn-save-alert-threshold", n_clicks=0,
                    className="btn btn-gold", style={"marginTop": "8px"}),
        html.Div(id="alert-threshold-msg", style={"marginTop": "10px", "fontSize": "11px"}),
    ])


def render_admin(auth: dict | None = None) -> html.Div:
    """Retourne la page Admin complète."""
    users = _db.list_users()
    audit = _db.get_audit_log(limit=100)

    return html.Div([
        # En-tête
        html.Div([
            html.Div("Administration", className="page-title"),
            html.Div("Gestion des utilisateurs et journal d'audit",
                     style={"fontSize": "11px", "color": "var(--muted)", "marginTop": "2px"}),
        ], style={"marginBottom": "24px"}),

        # Onglets
        dcc.Tabs(id="admin-tabs", value="users", className="beac-tabs", children=[
            dcc.Tab(label=f"Utilisateurs ({len(users)})", value="users",
                    className="beac-tab", selected_className="beac-tab--active"),
            dcc.Tab(label="Créer un compte",              value="create",
                    className="beac-tab", selected_className="beac-tab--active"),
            dcc.Tab(label=f"Audit ({len(audit)} événements)", value="audit",
                    className="beac-tab", selected_className="beac-tab--active"),
            dcc.Tab(label="Système", value="system",
                    className="beac-tab", selected_className="beac-tab--active"),
        ]),
        html.Div(id="admin-tab-content", style={"marginTop": "20px"}),

        # Données pré-chargées pour les onglets statiques
        dcc.Store(id="store-admin-users", data=users),
        dcc.Store(id="store-admin-audit", data=audit),
    ], className="page-content")
