"""Shell Dash : sidebar uniquement titlebar et menubar supprimés (gérés par Qt)."""
from dash import html, dcc
from .icons import (
    svg,
    ICO_FILE, ICO_PULSE, ICO_GRID, ICO_MODEL,
    ICO_SETTINGS, ICO_USERS, ICO_SLIDERS,
    ICO_CLOCK, ICO_BELL, ICO_SEARCH,
)
from config import PAYS_LABELS


def sidebar(active_page: str, slim: bool = False, running: bool = False,
            pays: str = "cameroun", volet: str = "Actif",
            progress: int = 0, progress_label: str = "",
            username: str = "analyste.beac",
            role: str = "analyste",
            unread_count: int = 0) -> html.Div:
    sb_cls = "sidebar slim" if slim else "sidebar"

    logo_content = [
        html.Img(src="/assets/logo_beac.jfif", className="sb-icon",
                 style={"width": "34px", "height": "34px", "objectFit": "contain",
                        "flexShrink": "0"}),
    ]
    if not slim:
        logo_content.append(
            html.Div(className="sb-wordmark", children=[
                html.Div("BEAC · CEMAC", className="sb-name"),
                html.Div("Anomalies", className="sb-sub"),
            ])
        )

    def _item(label, icon_d, page_id="", active=False):
        cls = "nav-item active" if active else "nav-item"
        children = [html.Div(className="nav-icon", children=[svg(icon_d, size=16)])]
        if not slim:
            children.append(html.Div(label, className="nav-label"))
        kw = {"id": f"nav-{page_id}", "n_clicks": 0} if page_id else {}
        if page_id:
            kw["role"] = "button"
            kw["tabIndex"] = 0
            if active:
                kw["aria-current"] = "page"
        return html.Div(className=cls, children=children, **kw)

    nav_items = [
        html.Div("Principal", className="nav-section-label"),
        _item("Données",      ICO_FILE,     "donnees",      active_page == "donnees"),
        _item("Analyse",      ICO_PULSE,    "analyse",      active_page == "analyse"),
        _item("Rapport",      ICO_GRID,     "rapport",      active_page == "rapport"),
        html.Div("Avancé", className="nav-section-label"),
        _item("Modèles",      ICO_MODEL,    "modeles",      active_page == "modeles"),
        _item("Optimisation", ICO_SLIDERS,  "optimisation", active_page == "optimisation"),
        _item("Historique",   ICO_CLOCK,    "historique",   active_page == "historique"),
    ]
    # Admin nav always rendered (for callback stability), hidden for non-admins
    admin_style = {} if role == "admin" else {"display": "none"}
    nav_items.append(
        html.Div(
            _item("Admin", ICO_USERS, "admin", active_page == "admin"),
            style=admin_style,
        )
    )

    # Footer VS Code style : Paramètres + Compte + badge running
    footer_children: list = []

    if running:
        footer_children.append(html.Div(
            style={"padding": "8px 6px", "borderBottom": "1px solid var(--border)", "marginBottom": "6px"},
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "5px", "marginBottom": "4px"},
                    children=[
                        html.Div(style={
                            "width": "6px", "height": "6px", "borderRadius": "50%",
                            "background": "var(--gold)", "animation": "spin 1s linear infinite",
                            "flexShrink": "0",
                        }),
                        html.Div("Analyse en cours", style={"fontSize": "9px", "letterSpacing": ".08em",
                                                             "textTransform": "uppercase", "color": "var(--muted2)"}),
                    ],
                ),
                html.Div(f"{PAYS_LABELS.get(pays, pays)} · {volet}",
                         style={"fontSize": "11px", "color": "var(--gold)"}),
            ]
        ))

    # Recherche + notifications (toujours visibles en bas, avant Paramètres)
    footer_children.append(html.Div(
        id="btn-open-search", n_clicks=0, className="nav-item",
        role="button", tabIndex=0,
        children=[
            html.Div(className="nav-icon", children=[svg(ICO_SEARCH, size=16)]),
            html.Div("Rechercher", className="nav-label") if not slim else None,
        ],
    ))
    badge_cls = "notif-badge" if unread_count else "notif-badge hidden"
    footer_children.append(html.Div(
        id="btn-notif-bell", n_clicks=0, className="nav-item",
        role="button", tabIndex=0, style={"position": "relative"},
        children=[
            html.Div(className="nav-icon", children=[
                svg(ICO_BELL, size=16),
                html.Span(str(unread_count) if unread_count else "",
                          id="notif-badge", className=badge_cls),
            ]),
            html.Div("Notifications", className="nav-label") if not slim else None,
        ],
    ))

    # Paramètres (toujours visible en bas, avant Compte)
    footer_children.append(_item("Paramètres", ICO_SETTINGS, "parametres",
                                  active_page == "parametres"))

    # Compte (toujours visible en bas, style VS Code)
    initials = "".join(p[0].upper() for p in username.split(".")[:2]) if username else "AN"
    footer_children.append(
        html.Div(
            id="nav-compte",
            n_clicks=0,
            role="button", tabIndex=0,
            **({"aria-current": "page"} if active_page == "compte" else {}),
            className="nav-item active" if active_page == "compte" else "nav-item",
            style={"gap": "8px"},
            children=[
                html.Div(initials, className="sb-avatar"),
                html.Div(username, className="nav-label sb-username") if not slim else None,
            ],
        )
    )

    return html.Div(className=sb_cls, children=[
        html.Div(className="sb-logo",
                 style={"justifyContent": "center", "padding": "12px"} if slim else {},
                 children=logo_content),
        html.Div(className="sb-nav", children=nav_items),
        html.Div(className="sb-footer", children=footer_children),
    ])


def shell_layout(page: str, main_content,
                 slim_sidebar: bool = False,
                 running: bool = False, pays: str = "cameroun",
                 volet: str = "Actif", progress: int = 0,
                 progress_label: str = "",
                 username: str = "analyste.beac",
                 role: str = "analyste",
                 unread_count: int = 0) -> html.Div:
    """Shell sans titlebar ni menubar Dash gérés par Qt."""
    return html.Div(
        style={"width": "100%", "height": "100vh", "display": "flex"},
        children=[
            sidebar(active_page=page, slim=slim_sidebar, running=running,
                    pays=pays, volet=volet, progress=progress,
                    progress_label=progress_label, username=username, role=role,
                    unread_count=unread_count),
            html.Div(className="main", id="main-content", children=[main_content]),
        ],
    )
