"""Callbacks Optimisation LOF* et BiVAT — algorithme génétique en thread."""
import time
import threading
import traceback
import plotly.io as pio
import dash
from dash import callback, Output, Input, State
from database import models as _db, close_connection
from callbacks.pipeline import _pipeline_state

_lof_optim_state: dict = {
    "running": False, "search": None, "result": None,
    "log": [], "progress": 0,
}
_bivat_optim_state: dict = {
    "running": False, "search": None, "result": None,
    "log": [], "progress": 0,
}


# ── Rendu des onglets ─────────────────────────────────────────────────────────

@callback(
    Output("optim-tab-content",    "children"),
    Input("optim-tabs",            "value"),
    State("store-optim-history",   "data"),
)
def render_optim_tab(tab, history):
    from pages.optimisation import _lof_tab, _bivat_tab, _history_tab, _aide_tab
    if tab == "lof":
        return _lof_tab()
    if tab == "bivat":
        return _bivat_tab()
    if tab == "history":
        return _history_tab(history or [])
    if tab == "aide":
        return _aide_tab()
    return dash.no_update


# ── LOF* — démarrage ──────────────────────────────────────────────────────────

@callback(
    Output("lof-optim-status",        "children"),
    Output("interval-lof-optim",      "disabled"),
    Input("btn-run-lof-optim",        "n_clicks"),
    State("lof-n-gen",    "value"),
    State("lof-pop-size", "value"),
    State("lof-elite",    "value"),
    State("lof-mut-rate", "value"),
    State("lof-cooling",  "value"),
    State("lof-patience", "value"),
    State("lof-min-div",  "value"),
    State("lof-opt-pays", "value"),
    State("lof-opt-volet","value"),
    State("store-auth",   "data"),
    prevent_initial_call=True,
)
def start_lof_optim(n, n_gen, pop_size, elite, mut_rate, cooling,
                    patience, min_div, pays, volet, auth):
    if not n or _lof_optim_state["running"]:
        return dash.no_update, dash.no_update

    # Reconstruire l'espace à partir des checkboxes — on lit via store-optim-lof-espace
    # (les checkboxes sont lues en State via pattern-matching dans le thread)
    from optimization.lof_search import ESPACE_LOF_DEFAULT, build_lof_fitness
    from heuristic_search import RechercheHeuristique
    from config import DATA_DIR

    user_id = (auth or {}).get("user_id")

    def _run():
        t0 = time.time()
        _lof_optim_state.update(running=True, result=None, log=[], progress=0)
        run_id = _db.create_optim_run(user_id, "lof", "genetic", pays, volet)
        try:
            fitness_fn = build_lof_fitness(
                data_dir = DATA_DIR,
                pays     = pays or "cameroun",
                volet    = volet or "Actif",
            )
            search = RechercheHeuristique(
                espace            = ESPACE_LOF_DEFAULT,
                fitness_fn        = fitness_fn,
                n_generations     = int(n_gen  or 15),
                taille_population = int(pop_size or 12),
                elite_size        = int(elite   or 2),
                mutation_rate     = float(mut_rate or 0.35),
                refroidissement   = float(cooling  or 0.97),
                min_diversite     = float(min_div  or 0.40),
                patience          = int(patience   or 5),
                maximize          = True,
            )
            _lof_optim_state["search"] = search
            result = search.fit()
            _lof_optim_state["result"] = result
            _lof_optim_state["progress"] = 100

            n_gen_done = len(search.history["generation"].unique()) if not search.history.empty else 0
            _db.finish_optim_run(
                run_id, "success",
                best_params  = result["params"],
                best_score   = result["score"],
                n_generations = n_gen_done,
                duration_ms  = int((time.time() - t0) * 1000),
            )
            _db.log_audit(user_id, "lof_optim_complete",
                          details={"run_id": run_id, "best_score": result["score"]})
        except Exception as e:
            _lof_optim_state["log"].append(f"[ERROR] {e}")
            _lof_optim_state["log"].append(traceback.format_exc())
            _db.finish_optim_run(run_id, "error", None, None, None,
                                 int((time.time() - t0) * 1000))
        finally:
            _lof_optim_state["running"] = False
            close_connection()

    threading.Thread(target=_run, daemon=True).start()
    return "Recherche LOF* en cours…", False


@callback(
    Output("interval-lof-optim", "disabled", allow_duplicate=True),
    Input("btn-stop-lof-optim",  "n_clicks"),
    prevent_initial_call=True,
)
def stop_lof_optim(n):
    _lof_optim_state["running"] = False
    return True


# ── LOF* — polling ────────────────────────────────────────────────────────────

@callback(
    Output("lof-optim-graphs",   "children"),
    Output("lof-optim-best",     "children"),
    Output("lof-optim-status",   "children", allow_duplicate=True),
    Output("interval-lof-optim", "disabled", allow_duplicate=True),
    Output("store-optim-history","data",     allow_duplicate=True),
    Input("interval-lof-optim",  "n_intervals"),
    prevent_initial_call=True,
)
def poll_lof_optim(n):
    search = _lof_optim_state.get("search")
    result = _lof_optim_state.get("result")
    running = _lof_optim_state["running"]

    if search is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Graphiques disponibles uniquement quand fit() est terminé
    if not running and result is not None:
        try:
            from dash import dcc as _dcc
            import plotly.io as _pio

            fig_conv = _pio.from_json(_pio.to_json(search.plot_convergence()))
            fig_space = _pio.from_json(_pio.to_json(search.plot_parameter_space()))
            fig_div  = _pio.from_json(_pio.to_json(search.plot_diversity()))
            fig_best = _pio.from_json(_pio.to_json(search.plot_best_evolution()))

            graphs = dash.html.Div([
                _dcc.Graph(figure=fig_conv,  config={"displayModeBar": False}),
                _dcc.Graph(figure=fig_best,  config={"displayModeBar": False}),
                _dcc.Graph(figure=fig_space, config={"displayModeBar": False}),
                _dcc.Graph(figure=fig_div,   config={"displayModeBar": False}),
            ])

            best_rows = [
                dash.html.Div([
                    dash.html.Span(f"{k} = ", style={"color": "var(--muted)", "fontSize": "11px"}),
                    dash.html.Span(str(v),    style={"color": "var(--gold)",  "fontSize": "11px",
                                                     "fontFamily": "Roboto Mono, monospace"}),
                ], style={"marginBottom": "4px"})
                for k, v in (result.get("params") or {}).items()
            ]
            best_card = dash.html.Div([
                dash.html.Div("Meilleurs hyperparamètres LOF*",
                              style={"fontSize": "10px", "textTransform": "uppercase",
                                     "letterSpacing": ".08em", "color": "var(--muted2)",
                                     "marginBottom": "10px"}),
                dash.html.Div(f"Score proxy = {result['score']:.4f}",
                              style={"fontSize": "13px", "fontWeight": "600",
                                     "color": "var(--green)", "marginBottom": "12px"}),
                *best_rows,
            ], style={"background": "var(--surface)", "border": "1px solid var(--border)",
                      "borderRadius": "6px", "padding": "14px"})

            history = _db.get_optim_history(limit=20)
            return graphs, best_card, "Recherche LOF* terminée.", True, history

        except Exception as exc:
            return (dash.html.Div(f"Erreur graphiques : {exc}", style={"color": "var(--red)"}),
                    dash.no_update, "Erreur lors de la génération des graphiques.", True,
                    dash.no_update)

    # Progression en cours
    journal = _lof_optim_state.get("log", [])
    log_els = [dash.html.Div(line, style={"fontSize": "10px", "color": "var(--muted)"})
               for line in journal[-5:]]
    return (dash.html.Div(log_els) if log_els else dash.no_update,
            dash.no_update, "Recherche LOF* en cours…", dash.no_update, dash.no_update)


# ── BiVAT — démarrage ─────────────────────────────────────────────────────────

@callback(
    Output("bivat-optim-status",   "children"),
    Output("interval-bivat-optim", "disabled"),
    Input("btn-run-bivat-optim",   "n_clicks"),
    State("bivat-n-gen",    "value"),
    State("bivat-pop-size", "value"),
    State("bivat-patience", "value"),
    State("store-auth",     "data"),
    prevent_initial_call=True,
)
def start_bivat_optim(n, n_gen, pop_size, patience, auth):
    if not n or _bivat_optim_state["running"]:
        return dash.no_update, dash.no_update

    lof_pipeline = _pipeline_state.get("lof_pipeline")
    if lof_pipeline is None:
        return "Erreur : Lancez d'abord une analyse LOF* (page Analyse).", True

    lof_key = _pipeline_state.get("lof_key")
    pays  = lof_key[0] if lof_key else "cameroun"
    volet = lof_key[1] if lof_key else "Actif"
    user_id = (auth or {}).get("user_id")

    from optimization.bivat_search import ESPACE_BIVAT_DEFAULT, build_bivat_fitness
    from heuristic_search import RechercheHeuristique

    def _run():
        t0 = time.time()
        _bivat_optim_state.update(running=True, result=None, log=[], progress=0)
        run_id = _db.create_optim_run(user_id, "bivat", "genetic", pays, volet)
        try:
            fitness_fn = build_bivat_fitness(lof_pipeline, pays=pays, volet=volet)
            search = RechercheHeuristique(
                espace            = ESPACE_BIVAT_DEFAULT,
                fitness_fn        = fitness_fn,
                n_generations     = int(n_gen    or 8),
                taille_population = int(pop_size or 8),
                elite_size        = 2,
                mutation_rate     = 0.40,
                refroidissement   = 0.95,
                min_diversite     = 0.35,
                patience          = int(patience or 3),
                maximize          = True,
            )
            _bivat_optim_state["search"] = search
            result = search.fit()
            _bivat_optim_state["result"] = result
            _bivat_optim_state["progress"] = 100

            n_gen_done = len(search.history["generation"].unique()) if not search.history.empty else 0
            _db.finish_optim_run(
                run_id, "success",
                best_params   = result["params"],
                best_score    = result["score"],
                n_generations = n_gen_done,
                duration_ms   = int((time.time() - t0) * 1000),
            )
            _db.log_audit(user_id, "bivat_optim_complete",
                          details={"run_id": run_id, "best_score": result["score"]})
        except Exception as e:
            _bivat_optim_state["log"].append(f"[ERROR] {e}")
            _bivat_optim_state["log"].append(traceback.format_exc())
            _db.finish_optim_run(run_id, "error", None, None, None,
                                 int((time.time() - t0) * 1000))
        finally:
            _bivat_optim_state["running"] = False
            close_connection()

    threading.Thread(target=_run, daemon=True).start()
    return "Recherche BiVAT en cours…", False


@callback(
    Output("interval-bivat-optim", "disabled", allow_duplicate=True),
    Input("btn-stop-bivat-optim",  "n_clicks"),
    prevent_initial_call=True,
)
def stop_bivat_optim(n):
    _bivat_optim_state["running"] = False
    return True


# ── BiVAT — polling ───────────────────────────────────────────────────────────

@callback(
    Output("bivat-optim-graphs",   "children"),
    Output("bivat-optim-best",     "children"),
    Output("bivat-optim-status",   "children", allow_duplicate=True),
    Output("interval-bivat-optim", "disabled", allow_duplicate=True),
    Input("interval-bivat-optim",  "n_intervals"),
    prevent_initial_call=True,
)
def poll_bivat_optim(n):
    search  = _bivat_optim_state.get("search")
    result  = _bivat_optim_state.get("result")
    running = _bivat_optim_state["running"]

    if search is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if not running and result is not None:
        try:
            from dash import dcc as _dcc
            import plotly.io as _pio

            fig_conv = _pio.from_json(_pio.to_json(search.plot_convergence()))
            fig_div  = _pio.from_json(_pio.to_json(search.plot_diversity()))
            fig_best = _pio.from_json(_pio.to_json(search.plot_best_evolution()))

            graphs = dash.html.Div([
                _dcc.Graph(figure=fig_conv,  config={"displayModeBar": False}),
                _dcc.Graph(figure=fig_best,  config={"displayModeBar": False}),
                _dcc.Graph(figure=fig_div,   config={"displayModeBar": False}),
            ])

            best_rows = [
                dash.html.Div([
                    dash.html.Span(f"{k} = ", style={"color": "var(--muted)", "fontSize": "11px"}),
                    dash.html.Span(str(v),    style={"color": "var(--gold)",  "fontSize": "11px",
                                                     "fontFamily": "Roboto Mono, monospace"}),
                ], style={"marginBottom": "4px"})
                for k, v in (result.get("params") or {}).items()
            ]
            best_card = dash.html.Div([
                dash.html.Div("Meilleurs hyperparamètres BiVAT",
                              style={"fontSize": "10px", "textTransform": "uppercase",
                                     "letterSpacing": ".08em", "color": "var(--muted2)",
                                     "marginBottom": "10px"}),
                dash.html.Div(f"Score proxy = {result['score']:.4f}",
                              style={"fontSize": "13px", "fontWeight": "600",
                                     "color": "var(--green)", "marginBottom": "12px"}),
                *best_rows,
            ], style={"background": "var(--surface)", "border": "1px solid var(--border)",
                      "borderRadius": "6px", "padding": "14px"})

            return graphs, best_card, "Recherche BiVAT terminée.", True

        except Exception as exc:
            return (dash.html.Div(f"Erreur : {exc}", style={"color": "var(--red)"}),
                    dash.no_update, "Erreur graphiques.", True)

    return dash.no_update, dash.no_update, "Recherche BiVAT en cours…", dash.no_update
