"""Page de connexion split-screen, sans carte encapsulante."""
import datetime
from dash import html, dcc
from layout.icons import svg, ICO_LOGIN, ICO_SHIELD


def render_login() -> html.Div:
    year = datetime.date.today().year
    return html.Div(
        id="login-screen",
        children=[
            # ── Canvas de dots pleine page ─────────────────────────────────
            html.Div(
                className="login-dots-canvas",
                children=[html.Div(className="login-dot") for _ in range(12)],
            ),

            # ── Panneau gauche branding BEAC ──────────────────────────────
            html.Div(
                className="login-brand-panel",
                children=[
                    html.Div(className="login-brand-stripe"),
                    html.Div(className="login-brand-grid"),
                    html.Img(
                        src="/assets/logo_beac.jfif",
                        alt="Logo BEAC",
                        className="login-brand-logo",
                    ),
                    html.Div("BEAC · CEMAC", className="login-brand-title"),
                    html.Div(
                        "Détection d'anomalies monétaires",
                        className="login-brand-subtitle",
                    ),
                    html.Div(className="login-brand-sep"),
                    html.Div(
                        className="login-brand-stats",
                        children=[
                            html.Div(
                                className="login-brand-stat",
                                children=[
                                    html.Div(className="login-brand-dot"),
                                    html.Span(txt),
                                ],
                            )
                            for txt in [
                                "6 États membres de la CEMAC",
                                "Modèles LOF*, BiVAT & KAN",
                            ]
                        ],
                    ),
                    html.Div(
                        className="login-brand-badge",
                        children=[
                            svg(ICO_SHIELD, size=11, stroke="var(--muted2)"),
                            html.Span("Accès réservé · COBAC v3.2"),
                        ],
                    ),
                ],
            ),

            # ── Panneau droit formulaire ───────────────────────────────────
            html.Div(
                className="login-form-panel",
                children=[
                    html.Div(
                        className="login-form-inner",
                        children=[
                            html.Div(
                                "Connexion au système",
                                className="login-form-heading",
                            ),

                            # Identifiant
                            html.Div(className="login-field-wrap", children=[
                                html.Label(
                                    "Identifiant",
                                    htmlFor="inp-login-user",
                                    className="login-field-label",
                                ),
                                dcc.Input(
                                    id="inp-login-user",
                                    type="text",
                                    value="",
                                    placeholder="identifiant.beac",
                                    className="login-input",
                                    autoFocus=False,
                                    autoComplete="username",
                                    n_submit=0,
                                ),
                            ]),

                            # Mot de passe
                            html.Div(className="login-field-wrap", children=[
                                html.Label(
                                    "Mot de passe",
                                    htmlFor="inp-login-pwd",
                                    className="login-field-label",
                                ),
                                dcc.Input(
                                    id="inp-login-pwd",
                                    type="password",
                                    value="",
                                    placeholder="••••••••",
                                    className="login-input",
                                    debounce=True,
                                    autoComplete="current-password",
                                    n_submit=0,
                                ),
                            ]),

                            # Erreur
                            html.Div(
                                id="login-error",
                                className="login-error",
                                role="alert",
                                **{"aria-live": "polite"},
                            ),

                            # Bouton
                            html.Button(
                                id="btn-login",
                                className="btn btn-solid-gold login-submit-btn",
                                children=[
                                    svg(ICO_LOGIN, size=14, sw=2.5),
                                    html.Span("Se connecter"),
                                ],
                                n_clicks=0,
                            ),

                            # Footer
                            html.Div(
                                className="login-footer",
                                children=[
                                    f"BEAC Anomaly Detector · v1.0.0 · {year}",
                                    html.Br(),
                                    "Direction de la Recherche et des Statistiques",
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )