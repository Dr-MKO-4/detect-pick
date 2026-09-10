"""BiVAT pipeline  full orchestration (SPEC §4, SPEC §5.2).

Steps:
    1. Receive pre-processed data from PipelineLOF (shared preprocessing)
    2. Build sliding windows
    3. Train BiVAT (Bi-Transformer + BiLSTM + VAE)
    4. Score: hybrid discriminant score s_discrim(t)
    5. Conformal Prediction (SSBC + EnbPI)
    6. SHAP explainability on top anomalies
    7. Return results dict for graphiques

Usage
-----
    from beac_lof.pipeline import PipelineLOF
    from bivat.pipeline import PipelineBiVAT

    lof = PipelineLOF()
    lof.fit("cameroun", "Actif")

    bivat = PipelineBiVAT()
    bivat.fit_from_lof(lof, "cameroun", "Actif")

    g = bivat.vers_graphiques("cameroun", "Actif")
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

import numpy as np
import pandas as pd

from bivat.architecture import BiVAT
from bivat import config as C
from bivat.training import train, save_checkpoint, load_checkpoint
from bivat.scoring import BiVATScorer
from bivat.explainability import explain_top_anomalies

logger = logging.getLogger(__name__)


class PipelineBiVAT:
    """
    BiVAT pipeline  one instance per (pays, volet).

    All intermediate results are stored as public attributes after fit.

    Public attributes after fit_from_lof()
    ---------------------------------------
    model             : BiVAT
    scores_train      : (T_train,)  NaN for first window-1 steps
    scores_cal        : (T_cal,)
    scores_test       : (T_test,)
    dates_train       : pd.DatetimeIndex
    dates_cal         : pd.DatetimeIndex
    dates_test        : pd.DatetimeIndex
    q_hat_95          : float   CP threshold at 95% coverage
    q_hat_90          : float   CP threshold at 90% coverage
    q_roll_95         : (T_test_valid,)
    q_roll_90         : (T_test_valid,)
    lo_95 / hi_95     : (T_test_valid,)   EnbPI intervals
    lo_90 / hi_90     : (T_test_valid,)
    anomalies_95      : (T_test,) bool
    shap_results      : dict (output of explain_top_anomalies)
    history           : list of epoch dicts
    feature_names     : list[str]
    """

    def __init__(self):
        self.model: BiVAT | None = None
        self.history: list[dict] = []
        self.scores_train = self.scores_cal = self.scores_test = None
        self.dates_train = self.dates_cal = self.dates_test = None
        self.q_hat_95 = self.q_hat_90 = None
        self.q_roll_95 = self.q_roll_90 = None
        self.lo_95 = self.hi_95 = self.lo_90 = self.hi_90 = None
        self.anomalies_95 = self.anomalies_90 = None
        self.shap_results: dict | None = None
        self.feature_names: list[str] = []
        self._pays = self._volet = ""

    # ── Main entry point ──────────────────────────────────────────────────────

    def fit_from_lof(
        self,
        lof_pipeline,           # beac_lof.pipeline.PipelineLOF instance
        pays: str,
        volet: str,
        epochs: int = C.EPOCHS,
        d_model: int = C.D_MODEL,
        n_heads: int = C.NHEAD,
        n_layers: int = C.N_ENCODER_LAYERS,
        window: int = C.WINDOW_SIZE,
        beta_kl: float = C.BETA,
        lambda_ad: float = C.LAMBDA_AD,
        lr: float = C.LR,
        train_end: str | None = None,
        test_start: str | None = None,
        cal_split: float | None = None,
        log_callback: Callable[[str], None] | None = None,
        force_retrain: bool = False,
        compute_shap: bool = True,
    ) -> "PipelineBiVAT":
        """
        Train (or load) BiVAT on normalised data from LOF pipeline.

        The LOF pipeline must have already run fit(pays, volet).
        BiVAT uses the normalised STL residuals as input.

        Parameters
        ----------
        d_model, n_heads, n_layers : architecture hyperparameters (passed to BiVAT)
        window                     : sliding window length (months)
        beta_kl                    : KL weight in the ELBO loss
        lambda_ad                  : poids de l'association discrepancy dans la loss
        lr                         : optimizer learning rate
        train_end, test_start      : bornes du split (défaut : C.TRAIN_END/C.TEST_START
                                      si None) — permet de reparamétrer la calibration
                                      depuis l'UI (page Modèles) sans toucher au code.
        cal_split                  : fraction du train prélevée en calibration si le
                                      calendrier train_end→test_start est trop court
                                      (défaut C.CAL_SPLIT si None)
        force_retrain              : if False and checkpoint exists, skip training
        compute_shap               : si False, saute l'étape SHAP DeepExplainer
                                      (coûteuse) — utile pour la recherche génétique
                                      d'hyperparamètres, qui n'exploite jamais
                                      shap_results dans sa fonction de fitness.
        """
        cal_split = C.CAL_SPLIT if cal_split is None else cal_split
        self._pays  = pays.lower()
        self._volet = volet
        key = (self._pays, volet)
        t_fit_start = time.time()

        logger.info("[BIVAT] fit_from_lof démarré  pays=%s  volet=%s", pays, volet)

        # ── 1. Get pre-processed data from LOF ────────────────────────────────
        residus: pd.DataFrame = lof_pipeline.residus.get(key)
        if residus is None or residus.empty:
            raise ValueError(f"LOF pipeline a-t-il fit({pays}, {volet}) ? Résidus manquants.")

        self.feature_names = residus.columns.tolist()
        X_full   = residus.values.astype(np.float32)          # (T, d)
        dates    = residus.index
        T, d     = X_full.shape
        logger.info("[BIVAT] Données LOF reçues  %d périodes × %d indicateurs  [%s → %s]",
                    T, d, str(dates[0])[:10], str(dates[-1])[:10])

        # ── 2. Split : train / cal / test ─────────────────────────────────────
        train_end  = pd.Timestamp(train_end or C.TRAIN_END)
        test_start = pd.Timestamp(test_start or C.TEST_START)

        train_mask = dates <= train_end
        cal_mask   = (dates > train_end) & (dates < test_start)
        test_mask  = dates >= test_start

        # Nombre de fenêtres de calibration exploitables si on utilise le
        # calendrier TRAIN_END/TEST_START tel quel (n_windows = n_mois - window + 1).
        n_cal_calendar = int(cal_mask.sum())
        n_cal_windows_calendar = max(0, n_cal_calendar - window + 1)

        if n_cal_windows_calendar >= C.MIN_CAL_WINDOWS:
            X_train = X_full[train_mask]
            X_cal   = X_full[cal_mask]
            X_test  = X_full[test_mask]
            self.dates_train = dates[train_mask]
            self.dates_cal   = dates[cal_mask]
            self.dates_test  = dates[test_mask]
        else:
            # Calendrier TRAIN_END/TEST_START insuffisant pour une calibration
            # conforme fiable (ex. 2 mois seulement entre les deux dates) : on
            # déplace la frontière et on prélève CAL_SPLIT de la fin du train
            # comme calibration, au lieu de padder avec la fin du train tout
            # en la recomptant dans le train (qui ne donnait qu'1 point de
            # calibration au final — Fig. E illisible).
            n_train_full = int(train_mask.sum())
            n_cal_take   = max(window + C.MIN_CAL_WINDOWS - 1,
                               int(n_train_full * cal_split))
            n_cal_take   = max(0, min(n_cal_take, n_train_full - window))  # garder du train
            logger.info(
                "[BIVAT] Calendrier cal insuffisant (%d fenêtre(s) sur %d mois) — "
                "repli sur les %d derniers mois du train comme calibration",
                n_cal_windows_calendar, n_cal_calendar, n_cal_take)

            X_train_full = X_full[train_mask]
            dates_train_full = dates[train_mask]
            X_train = X_train_full[:-n_cal_take] if n_cal_take > 0 else X_train_full
            X_cal   = X_train_full[-n_cal_take:] if n_cal_take > 0 else X_full[cal_mask]
            X_test  = X_full[test_mask]
            self.dates_train = dates_train_full[:-n_cal_take] if n_cal_take > 0 else dates_train_full
            self.dates_cal   = dates_train_full[-n_cal_take:] if n_cal_take > 0 else dates[cal_mask]
            self.dates_test  = dates[test_mask]

        # Garantir que X_cal contient au moins `window` périodes pour make_windows
        if len(X_cal) < window:
            needed = window - len(X_cal)
            take   = min(needed, len(X_train))
            logger.info("[BIVAT] Cal trop court (%d < window=%d) — complété avec %d périodes de fin de train",
                        len(X_cal), window, take)
            X_cal          = np.concatenate([X_train[-take:], X_cal])
            self.dates_cal = dates_train_full[-take:].append(self.dates_cal) if n_cal_windows_calendar < C.MIN_CAL_WINDOWS else dates[train_mask][-take:].append(self.dates_cal)

        if len(X_test) == 0:
            logger.warning("[BIVAT] Aucune donnée après TEST_START=%s  repli sur split 80/20", test_start)
            split = int(len(X_full) * 0.8)
            X_train = X_full[:split]
            X_test  = X_full[split:]
            X_cal   = X_train[-int(len(X_train) * cal_split):]
            self.dates_train = dates[:split]
            self.dates_test  = dates[split:]
            self.dates_cal   = self.dates_train[-len(X_cal):]

        d_in = X_full.shape[1]
        logger.info("[BIVAT] Split  train=%d (%s→%s)  cal=%d (%s→%s)  test=%d (%s→%s)",
                    len(X_train),
                    str(self.dates_train[0])[:10] if len(self.dates_train) else "?",
                    str(self.dates_train[-1])[:10] if len(self.dates_train) else "?",
                    len(X_cal),
                    str(self.dates_cal[0])[:10] if len(self.dates_cal) else "?",
                    str(self.dates_cal[-1])[:10] if len(self.dates_cal) else "?",
                    len(X_test),
                    str(self.dates_test[0])[:10] if len(self.dates_test) else "?",
                    str(self.dates_test[-1])[:10] if len(self.dates_test) else "?")
        logger.info("[BIVAT] Proportions  train=%.0f%%  cal=%.0f%%  test=%.0f%%",
                    100 * len(X_train) / T, 100 * len(X_cal) / T, 100 * len(X_test) / T)

        # ── 3. Train or load ──────────────────────────────────────────────────
        def _log_cb(msg: str):
            logger.info(msg)
            if log_callback:
                log_callback(msg)

        weights_path = os.path.join("models", f"bivat_{self._pays}_{self._volet.lower()}.pt")
        hp_record = {
            "epochs": epochs, "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "window": window, "beta_kl": beta_kl, "lambda_ad": lambda_ad, "lr": lr,
            "train_end": str(train_end.date()), "test_start": str(test_start.date()),
            "cal_split": cal_split,
            "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }

        def _save_hp_sidecar():
            try:
                import json
                with open(weights_path + ".json", "w", encoding="utf-8") as f:
                    json.dump(hp_record, f, indent=2)
            except Exception as e:
                logger.warning("[BIVAT] Sidecar hyperparamètres non écrit : %s", e)

        t_model = time.time()
        if not force_retrain and os.path.exists(weights_path):
            try:
                self.model = load_checkpoint(weights_path)
                _log_cb(f"[BIVAT] Poids chargés depuis {weights_path} (en {time.time()-t_model:.1f}s)")
            except Exception as _e:
                logger.warning("[BIVAT] Chargement poids échoué (%s)  ré-entraînement", _e)
                self.model = BiVAT(d_in=d_in, d_model=d_model, nhead=n_heads, n_layers=n_layers)
                self.history = train(self.model, X_train, epochs=epochs, lr=lr,
                                      window=window, beta=beta_kl, lambda_ad=lambda_ad, log_callback=_log_cb)
                save_checkpoint(self.model, weights_path)
                _save_hp_sidecar()
        else:
            if force_retrain:
                logger.info("[BIVAT] force_retrain=True  entraînement depuis zéro")
            else:
                logger.info("[BIVAT] Aucun checkpoint trouvé (%s)  entraînement depuis zéro", weights_path)
            self.model = BiVAT(d_in=d_in, d_model=d_model, nhead=n_heads, n_layers=n_layers)
            logger.info("[BIVAT] Architecture BiVAT instanciée  d_in=%d  d_model=%d  nhead=%d  n_layers=%d",
                        d_in, d_model, n_heads, n_layers)
            self.history = train(self.model, X_train, epochs=epochs, lr=lr,
                                  window=window, beta=beta_kl, lambda_ad=lambda_ad, log_callback=_log_cb)
            save_checkpoint(self.model, weights_path)
            _save_hp_sidecar()
        logger.info("[BIVAT] Modèle prêt en %.1fs", time.time() - t_model)

        # ── 4. Scoring + Conformal Prediction ────────────────────────────────
        logger.info("[BIVAT] >>> Étape scoring + Prédiction Conforme")
        t_score = time.time()
        scorer = BiVATScorer(self.model, X_train, X_cal, X_test, window=window)
        sc = scorer.run()

        self.scores_train = sc["scores_train"]
        self.scores_cal   = sc["scores_cal"]
        self.scores_test  = sc["scores_test"]
        self.q_hat_95     = sc["q_hat_95"]
        self.q_hat_90     = sc["q_hat_90"]
        self.q_roll_95    = sc["q_roll_95"]
        self.q_roll_90    = sc["q_roll_90"]
        self.lo_95 = sc["lo_95"]; self.hi_95 = sc["hi_95"]
        self.lo_90 = sc["lo_90"]; self.hi_90 = sc["hi_90"]
        self.anomalies_95 = sc["anomalies_95"]
        self.anomalies_90 = sc["anomalies_90"]
        logger.info("[BIVAT] Scoring terminé en %.1fs", time.time() - t_score)

        # ── Score hybride α·LOF*_norm + (1-α)·s_discrim_norm (spec §5) ──────
        self.hybrid_scores: np.ndarray | None = None
        try:
            lof_sc = lof_pipeline.scores_lof.get((pays.lower(), volet))
            if lof_sc is not None and self.scores_test is not None:
                test_idx = pd.DatetimeIndex(self.dates_test) if self.dates_test is not None else None
                if test_idx is not None:
                    lof_test = lof_sc.reindex(test_idx).values.astype(float)
                    bi_test  = self.scores_test.copy()
                    def _norm01(a: np.ndarray) -> np.ndarray:
                        lo, hi = np.nanmin(a), np.nanmax(a)
                        return (a - lo) / (hi - lo + 1e-12)
                    alpha = 0.5
                    self.hybrid_scores = alpha * _norm01(lof_test) + (1 - alpha) * _norm01(bi_test)
                    hs = self.hybrid_scores
                    n_anom_h = int(np.nansum(hs > 0.5))
                    logger.info("[BIVAT] Score hybride calculé (α=%.1f·LOF* + %.1f·BiVAT)  "
                                "μ=%.4f  max=%.4f  >0.5: %d points",
                                alpha, 1 - alpha,
                                float(np.nanmean(hs)), float(np.nanmax(hs)), n_anom_h)
                else:
                    logger.warning("[BIVAT] Hybrid score ignoré  dates_test vides")
            else:
                logger.warning("[BIVAT] Hybrid score ignoré  LOF* scores absents pour (%s, %s)", pays, volet)
        except Exception as _he:
            logger.warning("[BIVAT] Hybrid score non calculé : %s", _he)

        n_anom = int(np.sum(self.anomalies_95[~np.isnan(self.scores_test)])) if self.anomalies_95 is not None else "?"
        msg_scoring = (f"[BIVAT] Scoring terminé · q̂_95={self.q_hat_95:.4f} · "
                       f"q̂_90={self.q_hat_90:.4f} · {n_anom} anomalies à 95%")
        logger.info(msg_scoring)
        if log_callback:
            log_callback(msg_scoring)

        # ── 5. SHAP explainability ────────────────────────────────────────────
        if compute_shap:
            logger.info("[BIVAT] >>> Étape SHAP explainability (top-%d anomalies)", C.SHAP_TOP_N)
            t_shap = time.time()
            try:
                # Fenêtre de test uniquement (spec graphique.tex §Fig. F : "les
                # cinq mois les plus anomaux selon BiVAT sur la fenêtre de
                # test"). Utiliser np.concatenate([scores_train, scores_cal,
                # scores_test]) avec X_full était incorrect : X_cal réutilise
                # la fin de X_train quand la fenêtre de calibration est trop
                # courte (cf. section "Split" ci-dessus), donc les positions
                # du tableau concaténé ne correspondent plus aux positions
                # chronologiques de X_full au-delà de la frontière train/cal —
                # ce qui décalait top_t et cassait le calcul des dates/scores
                # dans figF_shap_top_anomalies (bivat/graphiques.py).
                self.shap_results = explain_top_anomalies(
                    self.model,
                    X_test,
                    self.scores_test,
                    self.feature_names,
                    window=window,
                )
                elapsed_shap = time.time() - t_shap
                logger.info("[BIVAT] SHAP terminé en %.1fs pour top-%d anomalies", elapsed_shap, C.SHAP_TOP_N)
                if log_callback:
                    log_callback(f"[BIVAT] SHAP calculé pour top-{C.SHAP_TOP_N} anomalies ({elapsed_shap:.0f}s)")
            except Exception as e:
                logger.warning("[BIVAT] SHAP échoué : %s", e, exc_info=True)
                self.shap_results = None
        else:
            logger.info("[BIVAT] SHAP explainability sautée (compute_shap=False)")
            self.shap_results = None

        logger.info("[BIVAT] fit_from_lof terminé en %.1fs total", time.time() - t_fit_start)
        return self

    # ── Graphiques ────────────────────────────────────────────────────────────

    def vers_graphiques(self, lof_pipeline=None):
        """Return BiVATGraphiques pre-loaded with all results."""
        from bivat.graphiques import BiVATGraphiques
        if lof_pipeline is None:
            logger.warning(
                "vers_graphiques(): lof_pipeline=None  Fig. D et Fig. G seront vides "
                "(elles nécessitent les scores LOF* pour la comparaison inter-modèles)."
            )
        return BiVATGraphiques(pipeline=self, lof_pipeline=lof_pipeline)

    def get_results(self) -> dict:
        """Serialisable results dict for Dash dcc.Store."""
        def _arr(a):
            return a.tolist() if isinstance(a, np.ndarray) else a

        return {
            "scores_test":  _arr(self.scores_test),
            "dates_test":   [str(d)[:10] for d in self.dates_test] if self.dates_test is not None else [],
            "q_hat_95":     self.q_hat_95,
            "q_hat_90":     self.q_hat_90,
            "anomalies_95": _arr(self.anomalies_95),
            "feature_names": self.feature_names,
            "shap_results": self.shap_results,
            "hybrid_scores": _arr(self.hybrid_scores) if self.hybrid_scores is not None else None,
            "pays":  self._pays,
            "volet": self._volet,
        }
