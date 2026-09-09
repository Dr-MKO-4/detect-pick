"""Callbacks pipeline LOF* et BiVAT — résultats persistés en SQLite + figures sur disque."""
import os
import time
import hashlib
import threading
import concurrent.futures
import traceback
import dash
from dash import callback, Output, Input, State
from config import DATA_DIR
from database import models, close_connection

# Incrémenter cette version invalide immédiatement tous les caches existants —
# à faire chaque fois que la logique du pipeline LOF (beac_lof/) change de
# façon à modifier les résultats produits pour des hyperparamètres identiques.
PIPELINE_CACHE_VERSION = "v1"


def _lof_cache_key(pays: str, volet: str) -> str | None:
    """Clé de cache pour un run LOF (pays, volet), hyperparamètres par défaut.

    Retourne None si le fichier source est introuvable (pas de mise en cache
    possible sans identifiant stable de version du dataset).
    """
    try:
        from beac_lof.config import FILE_MAP, MINPTS_LB, MINPTS_UB, PCA_VAR_TARGET
        fname = f"{FILE_MAP[pays.lower()]}_{volet}.xlsx"
        mtime = os.path.getmtime(os.path.join(DATA_DIR, fname))
    except (KeyError, OSError):
        return None
    raw = (f"{PIPELINE_CACHE_VERSION}:lof:{pays.lower()}:{volet}:{mtime}:"
           f"{MINPTS_LB}:{MINPTS_UB}:{PCA_VAR_TARGET}:iqr15")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cached_lof_run(cache_key: str | None) -> dict | None:
    """Retourne {run_id, figures_json} si un run identique est en cache, sinon None."""
    if not cache_key:
        return None
    run_id = models.get_cached_run_id(cache_key)
    if run_id is None:
        return None
    run = models.get_run_by_id(run_id)
    if not run or run.get("status") != "success":
        return None
    figs = models.load_figures_for_run(run_id)
    if not figs:
        return None
    return {"run_id": run_id, "figures_json": figs}


def _bivat_cache_key(pays: str, volet: str) -> str | None:
    """Clé de cache pour un run BiVAT (pays, volet) : invalide si le fichier
    source OU les poids BiVAT (modèle ré-entraîné/optimisé) ont changé."""
    try:
        from beac_lof.config import FILE_MAP
        fname = f"{FILE_MAP[pays.lower()]}_{volet}.xlsx"
        data_mtime = os.path.getmtime(os.path.join(DATA_DIR, fname))
        weights_path = os.path.join("models", f"bivat_{pays.lower()}_{volet.lower()}.pt")
        weights_mtime = os.path.getmtime(weights_path)
    except (KeyError, OSError):
        return None
    raw = f"{PIPELINE_CACHE_VERSION}:bivat:{pays.lower()}:{volet}:{data_mtime}:{weights_mtime}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cached_bivat_run(cache_key: str | None) -> dict | None:
    """Retourne {run_id, figures_json} si un run BiVAT identique est en cache."""
    if not cache_key:
        return None
    run_id = models.get_cached_run_id(cache_key)
    if run_id is None:
        return None
    run = models.get_run_by_id(run_id)
    if not run or run.get("status") != "success":
        return None
    figs = models.load_figures_for_run(run_id)
    if not figs:
        return None
    return {"run_id": run_id, "figures_json": figs}

_pipeline_state: dict = {
    "running": False, "log": [], "results": None,
    "progress": 0, "label": "",
    "lof_pipeline": None,
    "lof_graphiques": None,
    "lof_key": None,
    "run_id": None,
    "pending_bivat": None,   # (pays, volet, user_id) si BiVAT attend la fin de LOF
}
_bivat_state: dict = {
    "running": False, "log": [], "results": None,
    "progress": 0, "label": "",
    "run_id": None,
}


def _check_alert_threshold(pays: str, volet: str, model_label: str,
                           n_anom: int | None, run_id: int) -> None:
    """Notifie tous les utilisateurs si le nombre d'anomalies dépasse le seuil configuré."""
    if not n_anom:
        return
    try:
        threshold = int(models.get_app_setting("alert_threshold_anomalies", 5) or 5)
    except (TypeError, ValueError):
        threshold = 5
    if n_anom >= threshold:
        models.create_notification_for_all(
            "alert",
            f"Seuil d'anomalies dépassé — {pays}/{volet}",
            f"{n_anom} anomalies détectées ({model_label}) sur {pays}/{volet} "
            f"(seuil configuré : {threshold}).",
            run_id=run_id,
        )


# ── LOF* ──────────────────────────────────────────────────────────────────────

@callback(
    Output("store-pipeline-running", "data", allow_duplicate=True),
    Output("interval-pipeline",      "disabled", allow_duplicate=True),
    Input("btn-run-preprocess",      "n_clicks"),
    State("dd-pays",    "value"),
    State("dd-volet",   "value"),
    State("store-auth", "data"),
    prevent_initial_call=True,
)
def start_preprocessing(n, pays, volet, auth):
    """Lance uniquement la Phase 1 (chargement, imputation, MAD, STL, RPCA,
    ACP) sans le calcul LOF* — permet d'inspecter fig1/2/3/A avant de
    lancer l'analyse complète. Le prétraitement est mis en cache au niveau
    du pipeline (PipelineLOF._compute_upstream) : lancer ensuite l'analyse
    complète sur le même pays/volet le réutilise sans le recalculer."""
    if not n or _pipeline_state.get("running"):
        return dash.no_update, dash.no_update

    user_id = (auth or {}).get("user_id")

    def _run():
        t0 = time.time()
        _pipeline_state.update(running=True, log=[], results=None,
                               progress=0, run_id=None)

        def _log(msg: str):
            _pipeline_state["log"].append(msg)

        run_id = models.create_run(
            user_id, pays, volet, "lof",
            {"data_dir": DATA_DIR, "pays": pays, "volet": volet, "phase": "preprocess"},
        )
        _pipeline_state["run_id"] = run_id

        try:
            import plotly.io as pio
            from beac_lof.pipeline import PipelineLOF

            p = _pipeline_state.get("lof_pipeline")
            if p is None:
                p = PipelineLOF(data_dir=DATA_DIR)
                _pipeline_state["lof_pipeline"] = p

            _log("[INFO] Chargement XLSX…")
            _pipeline_state.update(label="Chargement XLSX…", progress=10)

            p.preprocess(pays, volet)
            _pipeline_state.update(label="Prétraitement terminé — génération figures…", progress=60)
            _log("[INFO] Prétraitement terminé (imputation · MAD · STL · RPCA · ACP).")

            g = p.vers_graphiques()
            figs_json: dict = {}
            phase1_tasks = [
                ("fig1", lambda: g.fig1_missing_data_map(pays, volet)),
                ("fig2", lambda: g.fig2_heatmap_correlations_stl(pays, volet)),
                ("fig3", lambda: g.fig3_decomposition_stl(pays, volet)),
                ("figA", lambda: g.figA_variance_rpca_temporelle(pays, volet)),
            ]
            for n_done, (fig_id, fn) in enumerate(phase1_tasks, start=1):
                try:
                    json_val = pio.to_json(fn())
                    figs_json[fig_id] = json_val
                    models.save_figure(run_id, fig_id, json_val)
                    _log(f"[FIG] {fig_id} sauvegardé")
                except Exception as fe:
                    _log(f"[WARN] {fig_id} : {fe}")
                _pipeline_state["progress"] = 60 + int(40 * n_done / len(phase1_tasks))

            results = {"figures_json": figs_json, "pays": pays, "volet": volet,
                      "run_id": run_id, "preprocess_only": True}
            _pipeline_state["results"] = results
            _pipeline_state["progress"] = 100

            duration_ms = int((time.time() - t0) * 1000)
            models.finish_run(run_id, "success", duration_ms, None, None)
            models.log_audit(user_id, "run_preprocess_complete",
                             details={"run_id": run_id, "pays": pays, "volet": volet})
            _log("[DONE] Prétraitement terminé — figures sauvegardées sur disque.")
        except Exception as e:
            _log(f"[ERROR] {e}")
            _log(traceback.format_exc())
            models.finish_run(run_id, "error", int((time.time() - t0) * 1000), None, None)
        finally:
            _pipeline_state["running"] = False
            close_connection()

    threading.Thread(target=_run, daemon=True).start()
    return True, False


@callback(
    Output("store-pipeline-running", "data"),
    Output("interval-pipeline",      "disabled"),
    Input("btn-run-pipeline",        "n_clicks"),
    State("dd-pays",    "value"),
    State("dd-volet",   "value"),
    State("dd-modele",  "value"),
    State("store-auth", "data"),
    prevent_initial_call=True,
)
def start_pipeline(n, pays, volet, modele, auth):
    if not n or modele == "bivat" or _pipeline_state.get("running"):
        return dash.no_update, dash.no_update

    user_id = (auth or {}).get("user_id")

    def _run():
        t0 = time.time()
        _pipeline_state.update(running=True, log=[], results=None,
                               progress=0, run_id=None)

        def _log(msg: str):
            _pipeline_state["log"].append(msg)

        cache_key = _lof_cache_key(pays, volet)
        cached    = _load_cached_lof_run(cache_key)
        run_id    = None
        is_new_run = False

        try:
            from beac_lof.pipeline import PipelineLOF

            p = _pipeline_state.get("lof_pipeline")
            if p is None:
                p = PipelineLOF(data_dir=DATA_DIR)
                _pipeline_state["lof_pipeline"] = p

            _log("[INFO] Chargement XLSX…")
            _pipeline_state.update(label="Chargement XLSX…", progress=5)

            _log("[INFO] Imputation MICE · MAD · STL · RPCA · LOF*…")
            _pipeline_state.update(label="Pipeline LOF en cours…", progress=20)

            # p.fit() reste nécessaire même en cas de cache figures : BiVAT et
            # la page Modèles ont besoin de residus/scores_lof/tau en mémoire.
            # Coût réduit grâce au cache amont module-level (beac_lof/pipeline.py).
            p.fit(pays, volet)
            _pipeline_state["lof_key"] = (pays, volet)
            _pipeline_state["progress"] = 65
            tau_val = p.tau.get((pays.lower(), volet), None)
            _log(f"[DONE] LOF* calculé · τ = {tau_val}")

            if cached:
                # ── Résultats identiques déjà en cache : figures réutilisées ──
                run_id = cached["run_id"]
                figs_json = cached["figures_json"]
                _pipeline_state["run_id"] = run_id
                _log(f"[INFO] Résultats identiques trouvés en cache (run #{run_id}) "
                     f"— figures non régénérées.")
                _pipeline_state.update(label="Résultats en cache", progress=95)

                results = p.get_results()
                results.update(figures_json=figs_json, pays=pays, volet=volet,
                               run_id=run_id)
                _pipeline_state["results"] = results
                models.log_audit(user_id, "run_lof_cache_hit",
                                 details={"run_id": run_id, "pays": pays, "volet": volet})
                _pipeline_state["progress"] = 100
                _log("[DONE] Analyse LOF* terminée (résultats en cache).")
            else:
                import plotly.io as pio

                is_new_run = True
                run_id = models.create_run(
                    user_id, pays, volet, modele,
                    {"data_dir": DATA_DIR, "pays": pays, "volet": volet},
                )
                _pipeline_state["run_id"] = run_id

                _pipeline_state.update(label="Génération figures…")
                g = p.vers_graphiques()
                _pipeline_state["lof_graphiques"] = g
                figs_json: dict = {}

                # Phase 1 (résultats clés) générée en priorité pour affichage immédiat
                priority_tasks = [
                    ("fig9",  lambda: g.fig9_boxplot_lof_annee(pays, volet)),
                    ("fig10", lambda: g.fig10_serie_temporelle_lof(pays, volet)),
                    ("fig12", lambda: g.fig12_bar_chart_indicateurs(pays, volet)),
                    ("fig13", lambda: g.fig13_heatmap_anomalies(pays, volet)),
                    ("fig8",  lambda: g.fig8_distribution_lof(pays, volet)),
                    ("fig14", lambda: g.fig14_heatmap_interpays(volet, animate=False)),
                    ("figB",  lambda: g.figB_concordance_actif_passif(pays)),
                ]
                secondary_tasks = [
                    ("fig1",  lambda: g.fig1_missing_data_map(pays, volet)),
                    ("fig2",  lambda: g.fig2_heatmap_correlations_stl(pays, volet)),
                    ("fig3",  lambda: g.fig3_decomposition_stl(pays, volet)),
                    ("figA",  lambda: g.figA_variance_rpca_temporelle(pays, volet)),
                    ("fig4",  lambda: g.fig4_scree_plot_rpca(pays, volet)),
                    ("fig5",  lambda: g.fig5_biplot_acp(pays, volet)),
                    ("fig6",  lambda: g.fig6_projection_umap(pays, volet)),
                    ("fig7",  lambda: g.fig7_profils_lof_minpts(pays, volet)),
                    ("fig11", lambda: g.fig11_superposition_lof_indicateur(pays, volet)),
                ]
                all_tasks = priority_tasks + secondary_tasks

                def _gen_one(task):
                    fig_id, fn = task
                    try:
                        return fig_id, pio.to_json(fn()), None
                    except Exception as fe:
                        return fig_id, None, str(fe)

                n_done = 0
                n_total = len(all_tasks)

                # Générer les figures prioritaires et publier un résultat partiel immédiat
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    for fig_id, json_val, err in pool.map(_gen_one, priority_tasks):
                        if json_val:
                            figs_json[fig_id] = json_val
                            models.save_figure(run_id, fig_id, json_val)
                            _log(f"[FIG] {fig_id} sauvegardé")
                        else:
                            _log(f"[WARN] {fig_id} : {err}")
                        n_done += 1
                        _pipeline_state["progress"] = 65 + int(20 * n_done / n_total)

                # Résultats partiels disponibles dès maintenant → UI peut s'afficher
                _pipeline_state["progress"] = 85
                results = p.get_results()
                results.update(figures_json=dict(figs_json), pays=pays, volet=volet,
                               run_id=run_id)
                _pipeline_state["results"] = results
                _log("[INFO] Figures prioritaires prêtes — affichage possible")

                # Générer les figures secondaires en arrière-plan
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    for fig_id, json_val, err in pool.map(_gen_one, secondary_tasks):
                        if json_val:
                            figs_json[fig_id] = json_val
                            models.save_figure(run_id, fig_id, json_val)
                            _log(f"[FIG] {fig_id} sauvegardé")
                        else:
                            _log(f"[WARN] {fig_id} : {err}")
                        n_done += 1
                        _pipeline_state["progress"] = 85 + int(10 * (n_done - len(priority_tasks)) / len(secondary_tasks))

                _pipeline_state["progress"] = 95
                results.update(figures_json=figs_json)
                _pipeline_state["results"] = results

                # Persistance anomaly_results
                lof_scores = results.get("lof_scores", {})
                anomaly_rows = [
                    {"period": str(k), "lof_score": float(v),
                     "is_anomaly": float(v) > float(tau_val or 0)}
                    for k, v in lof_scores.items()
                ]
                models.save_anomaly_rows(run_id, anomaly_rows)

                duration_ms = int((time.time() - t0) * 1000)
                n_anom = sum(1 for r in anomaly_rows if r["is_anomaly"])
                tau_mean = float(tau_val) if tau_val is not None else None
                models.finish_run(run_id, "success", duration_ms, n_anom, tau_mean)
                models.log_audit(user_id, "run_lof_complete",
                                 details={"run_id": run_id, "pays": pays, "volet": volet,
                                          "n_anomalies": n_anom})
                _check_alert_threshold(pays, volet, "LOF*", n_anom, run_id)
                if cache_key:
                    models.set_result_cache(cache_key, run_id)

                _pipeline_state["progress"] = 100
                _log("[DONE] Analyse LOF* terminée — figures sauvegardées sur disque.")

            # Auto-déclencher BiVAT si demandé
            pending = _pipeline_state.pop("pending_bivat", None)
            if pending:
                pb_pays, pb_volet, pb_uid = pending
                _log(f"[INFO] Lancement automatique BiVAT pour {pb_pays}/{pb_volet}…")
                _launch_bivat_thread(pb_pays, pb_volet, pb_uid)

        except Exception as e:
            _log(f"[ERROR] {e}")
            _log(traceback.format_exc())
            if is_new_run and run_id is not None:
                models.finish_run(run_id, "error", int((time.time() - t0) * 1000), None, None)
        finally:
            _pipeline_state["running"] = False
            close_connection()

    threading.Thread(target=_run, daemon=True).start()
    return True, False


@callback(
    Output("store-pipeline-running", "data", allow_duplicate=True),
    Input("btn-stop-pipeline",       "n_clicks"),
    prevent_initial_call=True,
)
def stop_pipeline(n):
    _pipeline_state["running"] = False
    return False


@callback(
    Output("store-results",          "data"),
    Output("store-pipeline-running", "data", allow_duplicate=True),
    Output("interval-pipeline",      "disabled", allow_duplicate=True),
    Output("store-progress",         "data", allow_duplicate=True),
    Output("store-progress-label",   "data", allow_duplicate=True),
    Input("interval-pipeline",       "n_intervals"),
    prevent_initial_call=True,
)
def poll_pipeline(n):
    progress = _pipeline_state.get("progress", 0)
    label    = _pipeline_state.get("label", "")
    if not _pipeline_state["running"] and _pipeline_state["results"] is not None:
        return _pipeline_state["results"], False, True, progress, label
    # store-pipeline-running est un Input de render_page (app.py) : le repousser
    # à chaque tick (même valeur True) reconstruit toute la page — sidebar
    # comprise — en boucle pendant le run, ce qui réinitialise les n_clicks des
    # boutons de nav à 0 et peut déclencher une navigation fantôme (cf. navigate()
    # dans callbacks/navigation.py). On ne le pousse donc que lorsqu'il change
    # réellement (passage à False, géré ci-dessus).
    return dash.no_update, dash.no_update, dash.no_update, progress, label


# ── BiVAT ─────────────────────────────────────────────────────────────────────

def _launch_bivat_thread(pays: str, volet: str, user_id):
    """Lance le thread BiVAT. Appelé par le callback OU après auto-LOF."""
    def _run():
        t0 = time.time()
        _bivat_state.update(running=True, log=[], results=None, progress=0, run_id=None)

        def _log(msg: str):
            _bivat_state["log"].append(msg)

        lof = _pipeline_state.get("lof_pipeline")
        if lof is None or _pipeline_state.get("lof_key") != (pays, volet):
            _log("[ERROR] LOF* non disponible pour ce pays/volet.")
            _bivat_state["running"] = False
            return

        run_id     = None
        is_new_run = False

        try:
            import plotly.io as pio
            from bivat.pipeline import PipelineBiVAT

            cache_key = _bivat_cache_key(pays, volet)
            cached    = _load_cached_bivat_run(cache_key)

            if cached:
                # Fichier source ET poids BiVAT inchangés depuis le dernier run
                # réussi : on retrouve le run existant (même figures) plutôt
                # que de tout recalculer (SHAP compris, l'étape la plus
                # coûteuse). fit_from_lof() reste nécessaire pour reconstruire
                # scores_test/q_hat en mémoire, mais recharge le checkpoint
                # .pt existant (force_retrain=False) et saute SHAP.
                run_id = cached["run_id"]
                _bivat_state["run_id"] = run_id
                _log(f"[INFO] Résultats BiVAT identiques trouvés en cache (run #{run_id}) "
                     f"— figures non régénérées.")
                _bivat_state.update(label="Résultats en cache", progress=40)

                b = PipelineBiVAT()
                b.fit_from_lof(lof, pays, volet, log_callback=_log, compute_shap=False)
                _bivat_state["progress"] = 90

                results = b.get_results()
                results.update(figures_json=cached["figures_json"], run_id=run_id)
                _bivat_state["results"] = results
                models.log_audit(user_id, "run_bivat_cache_hit",
                                 details={"run_id": run_id, "pays": pays, "volet": volet})
                _bivat_state["progress"] = 100
                _log("[DONE] BiVAT terminé (résultats en cache).")
            else:
                is_new_run = True
                run_id = models.create_run(user_id, pays, volet, "bivat", {"pays": pays, "volet": volet})
                _bivat_state["run_id"] = run_id

                _log("[INFO] Entraînement BiVAT…")
                _bivat_state.update(label="Entraînement BiVAT…", progress=5)

                b = PipelineBiVAT()
                b.fit_from_lof(lof, pays, volet, log_callback=_log)
                _bivat_state["progress"] = 70
                q = getattr(b, "q_hat_95", None)
                _log(f"[DONE] BiVAT entraîné · q̂₉₅ = {q:.4f}" if q else "[DONE] BiVAT entraîné")

                _bivat_state.update(label="Génération figures BiVAT…", progress=72)
                g = b.vers_graphiques(lof_pipeline=lof)
                figs_json: dict = {}
                bivat_tasks = [
                    ("figE",     g.figE_intervalles_cp_calibration),
                    ("figD",     lambda: g.figD_comparaison_lof_bivat(pays, volet)),
                    ("figE_bis", g.figE_bis_bivat_cp_test),
                    ("figF",     g.figF_shap_top_anomalies),
                    ("figG",     g.figG_confusion_lof_bivat),
                ]
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                    def _gen(task):
                        fid, fn = task
                        try:
                            return fid, pio.to_json(fn()), None
                        except Exception as fe:
                            return fid, None, str(fe)
                    for fig_id, json_val, err in pool.map(_gen, bivat_tasks):
                        if json_val:
                            figs_json[fig_id] = json_val
                            models.save_figure(run_id, fig_id, json_val)
                            _log(f"[INFO] {fig_id} sauvegardé")
                        else:
                            _log(f"[WARN] {fig_id} : {err}")
                        _bivat_state["progress"] = min(_bivat_state["progress"] + 5, 94)

                results = b.get_results()
                results.update(figures_json=figs_json, run_id=run_id)
                _bivat_state["results"] = results

                duration_ms = int((time.time() - t0) * 1000)
                models.finish_run(run_id, "success", duration_ms, results.get("n_anomalies"), None)
                models.log_audit(user_id, "run_bivat_complete",
                                 details={"run_id": run_id, "pays": pays, "volet": volet})
                _check_alert_threshold(pays, volet, "BiVAT", results.get("n_anomalies"), run_id)
                if cache_key:
                    models.set_result_cache(cache_key, run_id)
                _bivat_state["progress"] = 100
                _log("[DONE] BiVAT terminé — figures sauvegardées sur disque.")

        except Exception as e:
            _log(f"[ERROR] {e}")
            _log(traceback.format_exc())
            # Ne marquer "error" que le run qu'on vient de créer — jamais un
            # run en cache déjà marqué "success" (ses figures restent valides
            # même si fit_from_lof() échoue ensuite en relisant le checkpoint).
            if is_new_run and run_id is not None:
                models.finish_run(run_id, "error", int((time.time() - t0) * 1000), None, None)
        finally:
            _bivat_state["running"] = False
            close_connection()

    threading.Thread(target=_run, daemon=True).start()


@callback(
    Output("store-bivat-running", "data"),
    Output("interval-bivat",      "disabled"),
    Output("store-pipeline-running", "data", allow_duplicate=True),
    Output("interval-pipeline",      "disabled", allow_duplicate=True),
    Input("btn-run-pipeline",     "n_clicks"),
    State("dd-pays",    "value"),
    State("dd-volet",   "value"),
    State("dd-modele",  "value"),
    State("store-auth", "data"),
    prevent_initial_call=True,
)
def start_bivat_pipeline(n, pays, volet, modele, auth):
    if not n or modele != "bivat":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if _bivat_state.get("running"):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    user_id = (auth or {}).get("user_id")
    lof_ready = (
        _pipeline_state.get("lof_pipeline") is not None
        and _pipeline_state.get("lof_key") == (pays, volet)
    )

    if lof_ready:
        _launch_bivat_thread(pays, volet, user_id)
        return True, False, dash.no_update, dash.no_update
    else:
        # LOF absent ou mauvais pays/volet → lancer LOF d'abord, BiVAT suivra automatiquement
        _bivat_state["log"] = [f"[INFO] LOF* non disponible pour {pays}/{volet} — lancement automatique LOF* + BiVAT…"]
        _pipeline_state["pending_bivat"] = (pays, volet, user_id)
        if not _pipeline_state.get("running"):
            from beac_lof.pipeline import PipelineLOF
            from config import DATA_DIR
            p = _pipeline_state.get("lof_pipeline") or PipelineLOF(data_dir=DATA_DIR)
            _pipeline_state["lof_pipeline"] = p
            # Déclencher le pipeline LOF via un thread dédié
            def _auto_lof():
                import plotly.io as pio
                t0 = time.time()
                _pipeline_state.update(running=True, log=[], results=None, progress=0,
                                       label="LOF* (auto pour BiVAT)…", run_id=None)
                # Lecture seule du cache ici : ce flux ne génère qu'un sous-ensemble
                # de figures (5/14) — on ne l'écrit jamais dans le cache pour ne
                # pas faire croire à un futur run complet qu'un jeu partiel suffit.
                cache_key = _lof_cache_key(pays, volet)
                cached    = _load_cached_lof_run(cache_key)
                run_id = (cached["run_id"] if cached else
                          models.create_run(user_id, pays, volet, "lof",
                                            {"pays": pays, "volet": volet, "auto": True}))
                _pipeline_state["run_id"] = run_id
                try:
                    p.fit(pays, volet)
                    _pipeline_state["lof_key"] = (pays, volet)
                    _pipeline_state["progress"] = 65
                    tau_val = p.tau.get((pays.lower(), volet))

                    if cached:
                        _pipeline_state["log"].append(
                            f"[INFO] Résultats identiques trouvés en cache (run #{run_id}).")
                        figs_json = cached["figures_json"]
                        results = p.get_results()
                        results.update(figures_json=figs_json, pays=pays, volet=volet, run_id=run_id)
                        _pipeline_state["results"] = results
                        _pipeline_state["progress"] = 85
                    else:
                        g = p.vers_graphiques()
                        _pipeline_state["lof_graphiques"] = g
                        # Figures prioritaires uniquement
                        priority = [
                            ("fig9",  lambda: g.fig9_boxplot_lof_annee(pays, volet)),
                            ("fig10", lambda: g.fig10_serie_temporelle_lof(pays, volet)),
                            ("fig12", lambda: g.fig12_bar_chart_indicateurs(pays, volet)),
                            ("fig13", lambda: g.fig13_heatmap_anomalies(pays, volet)),
                            ("fig8",  lambda: g.fig8_distribution_lof(pays, volet)),
                        ]
                        figs_json = {}
                        def _gen(task):
                            fid, fn = task
                            try: return fid, pio.to_json(fn()), None
                            except Exception as fe: return fid, None, str(fe)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                            for fig_id, jv, err in pool.map(_gen, priority):
                                if jv: figs_json[fig_id] = jv; models.save_figure(run_id, fig_id, jv)
                        results = p.get_results()
                        results.update(figures_json=figs_json, pays=pays, volet=volet, run_id=run_id)
                        _pipeline_state["results"] = results
                        _pipeline_state["progress"] = 85
                        _sc = p.scores_lof.get((pays.lower(), volet))
                        n_anom = int((_sc > tau_val).sum()) if (_sc is not None and tau_val) else 0
                        models.finish_run(run_id, "success", int((time.time()-t0)*1000), n_anom, tau_val)
                except Exception as e:
                    _pipeline_state["log"].append(f"[ERROR] LOF* auto : {e}")
                    if not cached:
                        models.finish_run(run_id, "error", int((time.time()-t0)*1000), None, None)
                finally:
                    _pipeline_state["running"] = False
                    close_connection()
                    pending = _pipeline_state.pop("pending_bivat", None)
                    if pending:
                        _launch_bivat_thread(*pending)
            threading.Thread(target=_auto_lof, daemon=True).start()
            return True, False, True, False
        return True, False, dash.no_update, dash.no_update


@callback(
    Output("store-bivat-results",  "data"),
    Output("store-bivat-running",  "data", allow_duplicate=True),
    Output("interval-bivat",       "disabled", allow_duplicate=True),
    Output("store-progress",       "data", allow_duplicate=True),
    Output("store-progress-label", "data", allow_duplicate=True),
    Input("interval-bivat",       "n_intervals"),
    prevent_initial_call=True,
)
def poll_bivat(n):
    # store-progress / store-progress-label sont partagés avec poll_pipeline
    # (LOF) — c'est déjà le contrat attendu par la barre de progression
    # globale (callbacks/extras.py, callback "Animations globales"), qui
    # affiche le libellé "BiVAT" via store-bivat-running. Avant ce correctif,
    # seul poll_pipeline les alimentait : une exécution BiVAT seule (LOF déjà
    # en cache) n'affichait donc jamais aucune barre de progression.
    progress = _bivat_state.get("progress", 0)
    label    = _bivat_state.get("label", "")
    if not _bivat_state["running"] and _bivat_state["results"] is not None:
        return _bivat_state["results"], False, True, progress, label
    # Même raison que poll_pipeline() : store-bivat-running est un Input de
    # render_page, ne le repousser que lors d'un changement réel.
    return dash.no_update, dash.no_update, dash.no_update, progress, label


# ── Journal page Modèles ──────────────────────────────────────────────────────

@callback(
    Output("training-log",       "children"),
    Output("training-log-label", "children"),
    Output("lof-tau-display",    "children"),
    Output("lof-status-badge",   "children"),
    Output("lof-status-badge",   "className"),
    Input("interval-pipeline",   "n_intervals"),
    Input("interval-bivat",      "n_intervals"),
    prevent_initial_call=True,
)
def update_training_log(n_lof, n_bivat):
    state = _bivat_state if _bivat_state["running"] else _pipeline_state
    lines = state.get("log", [])

    if not lines:
        children = [dash.html.Div("En attente du lancement…",
                                  style={"color": "var(--muted2)"})]
    else:
        def _style(line: str) -> dict:
            if "[ERROR]" in line: return {"color": "var(--red)"}
            if "[DONE]"  in line or "[FIG]" in line: return {"color": "var(--gold)"}
            if "[INFO]"  in line: return {"color": "var(--green)"}
            return {}
        children = [dash.html.Div(line, style=_style(line)) for line in lines[-40:]]

    label = "En attente"
    if _pipeline_state["running"]:
        label = f"LOF* · {_pipeline_state.get('label','En cours…')} · {_pipeline_state.get('progress',0)}%"
    elif _bivat_state["running"]:
        label = f"BiVAT · {_bivat_state.get('label','En cours…')} · {_bivat_state.get('progress',0)}%"
    elif _pipeline_state.get("results"):
        label = "LOF* terminé"

    tau_display = "—"
    if _pipeline_state.get("results"):
        tau_map = getattr(_pipeline_state.get("lof_pipeline"), "tau", {})
        if tau_map:
            vals = list(tau_map.values())
            tau_display = f"{sum(vals)/len(vals):.3f}"

    if _pipeline_state.get("results"):
        badge_text, badge_cls = "Calibré",      "tag tag-green"
    elif _pipeline_state["running"]:
        badge_text, badge_cls = "En cours…",    "tag tag-gold"
    else:
        badge_text, badge_cls = "Non calibré",  "tag tag-grey"

    return children, label, tau_display, badge_text, badge_cls
