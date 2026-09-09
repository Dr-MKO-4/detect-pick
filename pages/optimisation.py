"""Page Optimisation heuristique des hyperparamètres LOF* et BiVAT."""
from dash import html, dcc
from optimization.lof_search  import ESPACE_LOF_DEFAULT
from optimization.bivat_search import ESPACE_BIVAT_DEFAULT


# ── Helpers UI ────────────────────────────────────────────────────────────────

def _desc(text: str) -> html.Div:
    return html.Div(text, style={
        "fontSize": "10px", "color": "var(--muted2)",
        "marginTop": "3px", "marginBottom": "10px",
        "lineHeight": "1.5",
    })


def _label(text: str) -> html.Label:
    return html.Label(text, style={
        "fontSize": "10px", "color": "var(--muted2)",
        "textTransform": "uppercase", "letterSpacing": ".08em",
        "marginBottom": "4px", "display": "block",
    })


def _range_input(param_id: str, default_vals: list, all_vals: list) -> html.Div:
    """Sélecteur multi-valeurs pour l'espace de recherche d'un paramètre."""
    options = [{"label": str(v), "value": v} for v in all_vals]
    return dcc.Checklist(
        id={"type": "opt-param", "index": param_id},
        options=options,
        value=default_vals,
        inline=True,
        inputStyle={"marginRight": "4px"},
        labelStyle={"marginRight": "12px", "fontSize": "11px"},
    )


def _section(title: str, children: list) -> html.Div:
    return html.Div([
        html.Div(title, style={
            "fontSize": "10px", "fontWeight": "600",
            "textTransform": "uppercase", "letterSpacing": ".08em",
            "color": "var(--muted)", "marginBottom": "12px",
            "borderBottom": "1px solid var(--border)", "paddingBottom": "6px",
        }),
        *children,
    ], style={"marginBottom": "24px"})


# ── Onglet LOF* ───────────────────────────────────────────────────────────────

def _lof_tab() -> html.Div:
    return html.Div([
        html.Div([
            # Colonne gauche : paramètres
            html.Div([
                _section("Espace de recherche", [
                    _label("k_min (borne inférieure MinPts)"),
                    _range_input("minpts_lb", [3, 5, 8], [2, 3, 5, 8, 10, 12, 15]),
                    _desc("Nombre minimal de voisins pour le calcul LOF. "
                          "Valeurs faibles → détection locale, sensible au bruit. "
                          "Recommandé : 3–10."),

                    _label("k_max (borne supérieure MinPts)"),
                    _range_input("minpts_ub", [15, 20, 25], [10, 15, 20, 25, 30, 40]),
                    _desc("Nombre maximal de voisins. LOF* prend le maximum sur [k_min, k_max]. "
                          "Valeurs élevées → détection globale mais coût O(n²·k). "
                          "Recommandé : 15–30."),

                    _label("Méthode de seuil τ"),
                    _range_input("tau_method", ["iqr15", "iqr20"], ["iqr15", "iqr20", "p95"]),
                    _desc("Méthode de calcul du seuil d'anomalie : "
                          "IQR×1.5 (Tukey classique, sensible), "
                          "IQR×2.0 (moins sensible, moins de faux-positifs), "
                          "P95 (95e percentile)."),

                    _label("Période STL"),
                    _range_input("stl_period", [12], [6, 12, 24]),
                    _desc("Période de la composante saisonnière STL (en mois). "
                          "12 = saisonnalité annuelle (recommandé pour données monétaires). "
                          "6 = semi-annuel, 24 = biennal."),

                    _label("Variance cible ACP (%)"),
                    _range_input("pca_var_target", [0.90], [0.80, 0.85, 0.90, 0.95]),
                    _desc("Fraction de variance conservée par l'ACP sur la composante L de RPCA. "
                          "Valeurs plus élevées → plus de dimensions, plus d'information mais "
                          "risque de sur-ajustement."),
                ]),

                _section("Paramètres de l'algorithme génétique", [
                    html.Div([
                        html.Div([
                            _label("Générations max"),
                            dcc.Input(id="lof-n-gen", type="number", value=15, min=3, max=100,
                                      className="form-input", style={"width": "80px"}),
                            _desc("Nombre maximal de générations. "
                                  "L'arrêt anticipé peut stopper avant si aucune amélioration."),
                        ], style={"flex": "1"}),
                        html.Div([
                            _label("Taille population"),
                            dcc.Input(id="lof-pop-size", type="number", value=12, min=4, max=50,
                                      className="form-input", style={"width": "80px"}),
                            _desc("Individus par génération. "
                                  "Plus grand → meilleure exploration, plus lent."),
                        ], style={"flex": "1"}),
                        html.Div([
                            _label("Élites"),
                            dcc.Input(id="lof-elite", type="number", value=2, min=1, max=10,
                                      className="form-input", style={"width": "80px"}),
                            _desc("Individus préservés sans modification entre générations."),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "gap": "16px"}),

                    html.Div([
                        html.Div([
                            _label("Taux de mutation initial"),
                            dcc.Input(id="lof-mut-rate", type="number", value=0.35,
                                      min=0.05, max=1.0, step=0.05,
                                      className="form-input", style={"width": "80px"}),
                            _desc("Probabilité de mutation par paramètre. "
                                  "Décroît avec le refroidissement."),
                        ], style={"flex": "1"}),
                        html.Div([
                            _label("Refroidissement"),
                            dcc.Input(id="lof-cooling", type="number", value=0.97,
                                      min=0.80, max=1.0, step=0.01,
                                      className="form-input", style={"width": "80px"}),
                            _desc("Facteur multiplicatif du taux de mutation par génération. "
                                  "1.0 = aucun, 0.97 = doux, 0.90 = fort."),
                        ], style={"flex": "1"}),
                        html.Div([
                            _label("Patience (early stop)"),
                            dcc.Input(id="lof-patience", type="number", value=5, min=1, max=30,
                                      className="form-input", style={"width": "80px"}),
                            _desc("Arrêt si aucune amélioration pendant N générations."),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "gap": "16px", "marginTop": "8px"}),

                    html.Div([
                        html.Div([
                            _label("Diversité minimale"),
                            dcc.Input(id="lof-min-div", type="number", value=0.40,
                                      min=0.10, max=1.0, step=0.05,
                                      className="form-input", style={"width": "80px"}),
                            _desc("Seuil de diversité (fraction individus uniques). "
                                  "En dessous : injection aléatoire (Hao & Solnon §4.3)."),
                        ], style={"flex": "1"}),
                        html.Div([
                            _label("Pays cible"),
                            dcc.Dropdown(
                                id="lof-opt-pays",
                                options=[
                                    {"label": "Cameroun",            "value": "cameroun"},
                                    {"label": "Centrafrique",        "value": "centrafrique"},
                                    {"label": "Congo",               "value": "congo"},
                                    {"label": "Gabon",               "value": "gabon"},
                                    {"label": "Guinée équatoriale",  "value": "guinee_equatoriale"},
                                    {"label": "Tchad",               "value": "tchad"},
                                ],
                                value="cameroun", clearable=False,
                                className="dd-input",
                            ),
                            _desc("Pays sur lequel évaluer la fitness."),
                        ], style={"flex": "1"}),
                        html.Div([
                            _label("Volet"),
                            dcc.Dropdown(
                                id="lof-opt-volet",
                                options=[
                                    {"label": "Actif",  "value": "Actif"},
                                    {"label": "Passif", "value": "Passif"},
                                ],
                                value="Actif", clearable=False,
                                className="dd-input",
                            ),
                            _desc("Volet du bilan monétaire."),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "gap": "16px", "marginTop": "8px"}),
                ]),

                html.Button("Lancer la recherche LOF*",
                            id="btn-run-lof-optim", n_clicks=0,
                            className="btn btn-gold",
                            style={"minWidth": "220px"}),
                html.Button("Arrêter",
                            id="btn-stop-lof-optim", n_clicks=0,
                            className="btn btn-red",
                            style={"minWidth": "100px", "marginLeft": "10px"}),
                html.Div(id="lof-optim-status",
                         style={"marginTop": "10px", "fontSize": "11px", "color": "var(--muted)"}),

            ], style={"flex": "1", "minWidth": "340px"}),

            # Colonne droite : graphiques
            html.Div([
                html.Div("Les graphiques apparaissent ici après la recherche.",
                         id="lof-optim-graphs",
                         style={"color": "var(--muted2)", "fontSize": "11px",
                                "padding": "20px 0"}),
                html.Div(id="lof-optim-best",
                         style={"marginTop": "16px"}),
            ], style={"flex": "1.2", "minWidth": "340px"}),

        ], style={"display": "flex", "gap": "32px", "alignItems": "flex-start"}),

        dcc.Interval(id="interval-lof-optim", interval=2_000, n_intervals=0, disabled=True),
    ])


# ── Onglet BiVAT ──────────────────────────────────────────────────────────────

def _bivat_tab() -> html.Div:
    return html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "La recherche BiVAT nécessite qu'une analyse LOF* ait été lancée sur la même "
                    "(pays, volet). Chaque évaluation entraîne un modèle BiVAT complet "
                    "prévoyez 2–5 min par individu.",
                    style={"fontSize": "11px", "color": "var(--gold)",
                           "background": "rgba(158,111,26,0.08)", "borderRadius": "4px",
                           "padding": "10px 14px", "marginBottom": "20px",
                           "border": "1px solid rgba(158,111,26,0.25)"},
                ),

                _section("Espace de recherche BiVAT", [
                    _label("Dimension modèle (d_model)"),
                    _range_input("d_model", [64, 128], [32, 64, 128, 256]),
                    _desc("Dimension des représentations internes du Bi-Transformer. "
                          "Valeurs plus élevées → plus expressif mais plus lent."),

                    _label("Nombre de têtes d'attention"),
                    _range_input("n_heads", [4], [2, 4, 8]),
                    _desc("Têtes d'attention multi-têtes. Doit diviser d_model."),

                    _label("Nombre de couches"),
                    _range_input("n_layers", [2], [1, 2, 3]),
                    _desc("Nombre de blocs Transformer empilés. "
                          "Plus = plus profond, risque de sur-ajustement sur courtes séries."),

                    _label("Taille de fenêtre (W)"),
                    _range_input("window", [12], [8, 12, 16, 24]),
                    _desc("Longueur de la fenêtre glissante (mois). "
                          "Impacte la capacité à capturer les dépendances temporelles."),

                    _label("Poids KL (β)"),
                    _range_input("beta_kl", [0.5, 1.0], [0.1, 0.5, 1.0, 2.0]),
                    _desc("Poids du terme KL dans le VAE (β-VAE). "
                          "Plus élevé → espace latent plus régulier mais reconstruction moins fidèle."),

                    _label("Taux d'apprentissage"),
                    _range_input("lr", [1e-3], [5e-4, 1e-3, 3e-3]),
                    _desc("Taux d'apprentissage Adam. "
                          "Valeurs trop élevées → instabilité. Trop faibles → convergence lente."),

                    _label("Epochs d'entraînement"),
                    _range_input("epochs", [50], [30, 50, 80]),
                    _desc("Epochs d'entraînement par évaluation. "
                          "Réduire pour accélérer la recherche (ex: 30), "
                          "augmenter pour une estimation plus fiable (ex: 80)."),
                ]),

                _section("Paramètres génétiques", [
                    html.Div([
                        html.Div([
                            _label("Générations max"),
                            dcc.Input(id="bivat-n-gen", type="number", value=8,
                                      min=2, max=30, className="form-input", style={"width": "80px"}),
                            _desc("Recommandé ≤ 10 pour BiVAT (coût élevé)."),
                        ], style={"flex": "1"}),
                        html.Div([
                            _label("Taille population"),
                            dcc.Input(id="bivat-pop-size", type="number", value=8,
                                      min=3, max=20, className="form-input", style={"width": "80px"}),
                            _desc("Recommandé ≤ 12 pour BiVAT."),
                        ], style={"flex": "1"}),
                        html.Div([
                            _label("Patience"),
                            dcc.Input(id="bivat-patience", type="number", value=3,
                                      min=1, max=15, className="form-input", style={"width": "80px"}),
                            _desc("Arrêt anticipé si N générations sans amélioration."),
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "gap": "16px"}),
                ]),

                html.Button("Lancer la recherche BiVAT",
                            id="btn-run-bivat-optim", n_clicks=0,
                            className="btn btn-gold",
                            style={"minWidth": "220px"}),
                html.Button("Arrêter",
                            id="btn-stop-bivat-optim", n_clicks=0,
                            className="btn btn-red",
                            style={"minWidth": "100px", "marginLeft": "10px"}),
                html.Div(id="bivat-optim-status",
                         style={"marginTop": "10px", "fontSize": "11px", "color": "var(--muted)"}),

            ], style={"flex": "1", "minWidth": "340px"}),

            html.Div([
                html.Div("Les graphiques apparaissent ici après la recherche.",
                         id="bivat-optim-graphs",
                         style={"color": "var(--muted2)", "fontSize": "11px",
                                "padding": "20px 0"}),
                html.Div(id="bivat-optim-best",
                         style={"marginTop": "16px"}),
            ], style={"flex": "1.2", "minWidth": "340px"}),

        ], style={"display": "flex", "gap": "32px", "alignItems": "flex-start"}),

        dcc.Interval(id="interval-bivat-optim", interval=3_000, n_intervals=0, disabled=True),
    ])


# ── Historique ────────────────────────────────────────────────────────────────

def _history_tab(history: list[dict]) -> html.Div:
    if not history:
        return html.Div("Aucune recherche enregistrée.",
                        style={"color": "var(--muted2)", "fontSize": "11px", "padding": "20px 0"})
    rows = []
    for h in history:
        best_params = ""
        try:
            import json as _j
            bp = _j.loads(h.get("best_params") or "{}")
            best_params = " · ".join(f"{k}={v}" for k, v in bp.items())[:80]
        except Exception:
            pass
        rows.append(html.Tr([
            html.Td(h.get("created_at", "")[:16],
                    style={"fontSize": "10px", "fontFamily": "Roboto Mono, monospace",
                           "color": "var(--muted)", "whiteSpace": "nowrap"}),
            html.Td(h.get("model", "").upper(), style={"fontSize": "11px"}),
            html.Td(f"{h.get('pays','')} · {h.get('volet','')}", style={"fontSize": "11px"}),
            html.Td(f"{h.get('best_score', 0):.4f}" if h.get("best_score") is not None else "—",
                    style={"fontSize": "11px", "color": "var(--gold)",
                           "fontFamily": "Roboto Mono, monospace"}),
            html.Td(str(h.get("n_generations", "—")), style={"fontSize": "11px"}),
            html.Td(best_params, style={"fontSize": "10px", "color": "var(--muted2)"}),
        ]))
    return html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("Date"),
                html.Th("Modèle"),
                html.Th("Périmètre"),
                html.Th("Meilleur score"),
                html.Th("Générations"),
                html.Th("Meilleurs paramètres"),
            ], style={"fontSize": "10px", "textTransform": "uppercase",
                      "letterSpacing": ".08em", "color": "var(--muted2)"})),
            html.Tbody(rows),
        ], className="data-table", style={"width": "100%"}),
    ])


# ── Onglet Aide ───────────────────────────────────────────────────────────────

def _aide_tab() -> html.Div:
    def _h2(text: str) -> html.Div:
        return html.Div(text, style={
            "fontSize": "13px", "fontWeight": "600", "color": "var(--fg)",
            "marginTop": "24px", "marginBottom": "8px",
            "borderBottom": "1px solid var(--border)", "paddingBottom": "6px",
        })
    def _p(text: str) -> html.P:
        return html.P(text, style={"fontSize": "12px", "color": "var(--muted)",
                                   "lineHeight": "1.7", "marginBottom": "10px"})
    def _ul(items: list[str]) -> html.Ul:
        return html.Ul([
            html.Li(it, style={"fontSize": "12px", "color": "var(--muted)",
                               "lineHeight": "1.7", "marginBottom": "4px"})
            for it in items
        ], style={"paddingLeft": "20px", "marginBottom": "12px"})

    return html.Div([
        html.Div("Aide Optimisation heuristique des hyperparamètres",
                 style={"fontSize": "16px", "fontWeight": "600", "marginBottom": "8px"}),

        _h2("Principe général"),
        _p("L'optimisation heuristique cherche les meilleurs hyperparamètres des modèles LOF* "
           "et BiVAT sans disposer de vérité terrain (données non étiquetées). "
           "Elle s'appuie sur une fonction de fitness proxy qui évalue indirectement la "
           "qualité de la détection."),

        _h2("Algorithme génétique (Hao & Solnon, 2024)"),
        _p("L'algorithme maintient une population d'individus (jeux d'hyperparamètres) "
           "et les fait évoluer par :"),
        _ul([
            "Sélection élitisme (meilleurs préservés) + roulette proportionnelle + tournoi.",
            "Croisement monopont échange d'une portion des paramètres entre deux parents.",
            "Mutation adaptative modification aléatoire avec un taux décroissant "
            "(analogie recuit simulé) : exploration en début, exploitation en fin.",
            "Contrôle de diversité injection d'individus aléatoires si la population "
            "converge trop vite (fraction d'individus uniques < min_diversite).",
            "Composante EDA modèle probabiliste sur les meilleurs individus pour guider "
            "la génération d'enfants vers les régions prometteuses.",
        ]),

        _h2("Fonction de fitness LOF*"),
        _p("Trois composantes (proxy sans étiquettes) :"),
        _ul([
            "Rang des périodes documentées (50 %) : COVID mars 2020–déc. 2021, choc "
            "pétrolier 2014–2016, crise CEMAC 2017–2019. Les hyperparamètres qui font "
            "remonter ces périodes parmi les scores les plus élevés sont récompensés.",
            "Rappel sur anomalies synthétiques (40 %) : 10 % de points bruités (+3σ) "
            "sont injectés. Le taux de détection mesure la sensibilité du pipeline.",
            "Pénalité faux-positifs (10 %) : si le taux d'anomalies dépasse 30 %, "
            "un malus est appliqué pour éviter les pipelines trop sensibles.",
        ]),

        _h2("Fonction de fitness BiVAT"),
        _p("Deux composantes :"),
        _ul([
            "Cohérence LOF*–BiVAT (70 %) : fraction des anomalies détectées par LOF* "
            "qui sont aussi signalées par BiVAT. Mesure la concordance entre les deux modèles.",
            "Pénalité faux-positifs (30 %) : même logique que LOF*.",
        ]),

        _h2("Lecture des graphiques"),
        _ul([
            "Convergence : courbe bleue = meilleur score de la génération, "
            "orange pointillé = score moyen, bande = ±1σ. "
            "Idéalement la courbe bleue monte et se stabilise.",
            "Évolution du meilleur : courbe monotone croissante du meilleur score cumulatif. "
            "Un plateau précoce peut indiquer que l'espace est trop petit ou la population trop réduite.",
            "Espace des paramètres : boîtes à moustaches par valeur de chaque hyperparamètre. "
            "Les valeurs avec des boîtes hautes sont associées à de meilleures performances.",
            "Diversité & mutation : la diversité doit rester au-dessus du seuil min_diversite. "
            "Le taux de mutation doit décroître progressivement vers son plancher.",
        ]),

        _h2("Recommandations pratiques"),
        _ul([
            "LOF* : commencer avec 10–15 générations et population 12. "
            "L'évaluation est rapide (quelques secondes) ; une recherche complète prend 2–5 min.",
            "BiVAT : limiter à 6–8 générations et population 8. "
            "Chaque évaluation prend 2–5 min une recherche complète peut durer 1–3 h.",
            "Si la convergence est trop rapide (plateau dès la gen 2), "
            "augmenter la diversité minimale ou le taux de mutation initial.",
            "Si la fitness ne dépasse jamais 0.50, vérifier que les données XLSX couvrent "
            "bien les périodes documentées (COVID, choc pétrolier).",
        ]),

        _h2("Limites"),
        _p("La fonction proxy n'est pas équivalente à une évaluation sur données étiquetées. "
           "Les hyperparamètres optimaux selon la proxy peuvent ne pas être les meilleurs "
           "en production. Les résultats doivent être validés manuellement en comparant "
           "les anomalies détectées avec la connaissance métier."),
    ], style={"maxWidth": "800px", "paddingBottom": "40px"})


# ── Page complète ─────────────────────────────────────────────────────────────

def render_optimisation(history: list[dict] | None = None) -> html.Div:
    from database import models as _db
    history = history or _db.get_optim_history(limit=20)

    return html.Div([
        html.Div([
            html.Div("Optimisation heuristique", className="page-title"),
            html.Div("Recherche des meilleurs hyperparamètres LOF* et BiVAT par algorithme génétique",
                     style={"fontSize": "11px", "color": "var(--muted)", "marginTop": "2px"}),
        ], style={"marginBottom": "24px"}),

        dcc.Tabs(id="optim-tabs", value="lof", className="beac-tabs", children=[
            dcc.Tab(label="LOF* Recherche",  value="lof",
                    className="beac-tab", selected_className="beac-tab--active"),
            dcc.Tab(label="BiVAT Recherche", value="bivat",
                    className="beac-tab", selected_className="beac-tab--active"),
            dcc.Tab(label=f"Historique ({len(history)})", value="history",
                    className="beac-tab", selected_className="beac-tab--active"),
            dcc.Tab(label="Aide", value="aide",
                    className="beac-tab", selected_className="beac-tab--active"),
        ]),
        html.Div(id="optim-tab-content", style={"marginTop": "20px"}),
        dcc.Store(id="store-optim-history", data=history),

    ], className="page-content")
