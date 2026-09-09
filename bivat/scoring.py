"""BiVAT scoring  hybrid score + Conformal Prediction (SPEC §4.1 D2, D4).

Hybrid score (Xu et al. 2022):
    s_discrim(t) = Reconstruction_error(t) × Association_Discrepancy(t)

Conformal Prediction:
    SSBC (Split Conformal + Bootstrapped Calibration) for N < 200
    EnbPI (rolling-origin adaptation) for non-stationarity
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from bivat.architecture import BiVAT
from bivat.training import make_windows
from bivat.utils import assoc_discrepancy as _assoc_discrepancy_batch
from bivat import config as C
from utils.hardware import DEVICE

logger = logging.getLogger(__name__)


# ── Hybrid score ──────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_scores(
    model: BiVAT,
    X: np.ndarray,
    window: int = C.WINDOW_SIZE,
) -> np.ndarray:
    """
    Compute hybrid discriminant score s_discrim for each time step.

    Windows slide over X (T×d); score for each window is assigned to the
    last time step of that window (causal assignment).

    Returns
    -------
    scores : (T,) array  NaN for the first (window-1) time steps
    """
    model.eval()
    model.to(DEVICE)
    windows = make_windows(X, window).to(DEVICE)   # (n, window, d)
    n, W, d = windows.shape

    logger.info("[BiVATScorer] compute_scores  %d fenêtres (%d×%d×%d), device=%s",
                n, n, W, d, DEVICE)

    # Batch inference to avoid OOM
    batch = max(1, 256 // W)
    rec_errors = []
    ads        = []
    n_batches  = (n + batch - 1) // batch
    log_every  = max(1, n_batches // 4)

    for i in range(0, n, batch):
        batch_idx = i // batch
        xb  = windows[i: i + batch]
        out = model(xb)
        re  = F.mse_loss(out["x_hat"], xb, reduction="none").mean(dim=[1, 2])  # (B,)
        ad  = _assoc_discrepancy_batch(out["attn"])                              # (B,)
        rec_errors.append(re.cpu().numpy())
        ads.append(ad.cpu().numpy())

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
            logger.debug("[BiVATScorer] batch %d/%d  RE moy=%.5f  AD moy=%.5f",
                         batch_idx + 1, n_batches,
                         float(re.mean().item()), float(ad.mean().item()))

    rec_errors = np.concatenate(rec_errors)          # (n_windows,)
    ads        = np.concatenate(ads)
    raw_scores = rec_errors * ads

    # Map window score to last time step
    T = len(X)
    scores = np.full(T, np.nan)
    for i, s in enumerate(raw_scores):
        t = i + window - 1   # last index of the window
        scores[t] = s

    valid = scores[~np.isnan(scores)]
    logger.info("[BiVATScorer] Scores calculés  %d valides/%d · μ=%.5f  σ=%.5f  max=%.5f  "
                "p75=%.5f  p95=%.5f  p99=%.5f",
                len(valid), T,
                float(np.mean(valid)), float(np.std(valid)),
                float(np.max(valid)),
                float(np.quantile(valid, 0.75)),
                float(np.quantile(valid, 0.95)),
                float(np.quantile(valid, 0.99)))
    logger.info("[BiVATScorer] Décomposition  RE: μ=%.5f  max=%.5f | AD: μ=%.5f  max=%.5f",
                float(np.mean(rec_errors)), float(np.max(rec_errors)),
                float(np.mean(ads)), float(np.max(ads)))
    return scores


# ── SSBC  Split Conformal + Bootstrapped Calibration ────────────────────────

class SSBCConformal:
    """
    Split Conformal with Bootstrap Calibration (SSBC) for N < 200.

    Guarantee: Pr(score(x_t) ≤ q̂_{1-α}) ≥ 1 - α
    SSBC corrects the quantile for small calibration sets.

    Parameters
    ----------
    alpha : float   desired miscoverage level (e.g. 0.05 for 95 % coverage)
    """

    def __init__(self, alpha: float = C.ALPHA_LOW):
        self.alpha  = alpha
        self.q_hat  = None          # calibrated threshold
        self.cal_scores: np.ndarray | None = None

    def calibrate(self, scores_cal: np.ndarray) -> float:
        """
        Compute q̂_{1-α} with finite-sample correction:
            q̂ = quantile(scores_cal, (1-α)(1 + 1/n))
        where n = len(scores_cal) (Vovk et al. 2005, Angelopoulos & Bates 2021).
        """
        scores_cal = scores_cal[~np.isnan(scores_cal)]
        n = len(scores_cal)
        if n == 0:
            logger.warning("[SSBC α=%.2f] Calibration impossible  aucun score valide", self.alpha)
            self.q_hat = float("nan")
            return self.q_hat
        level = min((1 - self.alpha) * (1 + 1 / n), 1.0)
        self.q_hat     = float(np.quantile(scores_cal, level))
        self.cal_scores = scores_cal
        logger.info("[SSBC α=%.2f] Calibration  n_cal=%d · niveau=%.4f · q̂=%.5f "
                    "(scores cal: μ=%.5f  p50=%.5f  max=%.5f)",
                    self.alpha, n, level, self.q_hat,
                    float(np.mean(scores_cal)),
                    float(np.median(scores_cal)),
                    float(np.max(scores_cal)))
        return self.q_hat

    def intervals(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) CP interval at level α around each score."""
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")
        lower = scores - self.q_hat
        upper = scores + self.q_hat
        return lower, upper

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Return boolean anomaly flags: score > q̂."""
        return scores > self.q_hat


# ── EnbPI  rolling-origin adaptation ────────────────────────────────────────

class EnbPI:
    """
    Ensemble Batch Prediction Intervals (EnbPI) for online conformal adaptation.

    At each new time step t, the quantile is updated using the residuals from
    the last `lookback` steps. This handles non-stationarity by tracking
    recent errors (e.g. during COVID regime shift).

    Reference: Xu & Xie (2021)  Conformal Prediction Interval for Dynamic
    Time-Series.
    """

    def __init__(self, alpha: float = C.ALPHA_LOW, lookback: int = 24):
        self.alpha    = alpha
        self.lookback = lookback
        self._residuals: list[float] = []

    def update(self, score: float) -> None:
        self._residuals.append(score)
        if len(self._residuals) > self.lookback:
            self._residuals.pop(0)

    def q_hat(self) -> float:
        if not self._residuals:
            return float("nan")
        arr = np.array(self._residuals)
        n   = len(arr)
        lvl = min((1 - self.alpha) * (1 + 1 / n), 1.0)
        return float(np.quantile(arr, lvl))

    def rolling_intervals(
        self, scores: np.ndarray, seed_scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute rolling-origin quantiles and intervals.

        Parameters
        ----------
        scores      : test scores  (T_test,)
        seed_scores : calibration scores to seed the buffer

        Returns
        -------
        q_hats  : (T_test,)  quantile at each step
        lower   : (T_test,)
        upper   : (T_test,)
        """
        for s in seed_scores[~np.isnan(seed_scores)]:
            self.update(float(s))

        q_hats = []
        for s in scores:
            q = self.q_hat()
            q_hats.append(q)
            if not np.isnan(s):
                self.update(float(s))

        q_hats = np.array(q_hats)
        return q_hats, scores - q_hats, scores + q_hats


# ── Convenience wrapper ───────────────────────────────────────────────────────

class BiVATScorer:
    """
    Full scoring wrapper: compute scores, calibrate, return thresholds.

    Usage
    -----
    scorer = BiVATScorer(model, X_train, X_cal, X_test)
    results = scorer.run()
    # results["scores_test"], results["q_hat_90"], etc.
    """

    def __init__(self, model: BiVAT,
                 X_train: np.ndarray,
                 X_cal:   np.ndarray,
                 X_test:  np.ndarray,
                 window:  int = C.WINDOW_SIZE):
        self.model   = model
        self.X_train = X_train
        self.X_cal   = X_cal
        self.X_test  = X_test
        self.window  = window

    def run(self) -> dict:
        logger.info("[BiVATScorer] Démarrage scoring complet  train=%d  cal=%d  test=%d",
                    len(self.X_train), len(self.X_cal), len(self.X_test))

        logger.info("[BiVATScorer] >>> Scoring train")
        scores_train = compute_scores(self.model, self.X_train, window=self.window)
        logger.info("[BiVATScorer] >>> Scoring calibration")
        scores_cal   = compute_scores(self.model, self.X_cal, window=self.window)
        logger.info("[BiVATScorer] >>> Scoring test")
        scores_test  = compute_scores(self.model, self.X_test, window=self.window)

        # SSBC at 95% and 90%
        logger.info("[BiVATScorer] >>> Calibration SSBC (Prédiction Conforme Splitée)")
        ssbc_95 = SSBCConformal(alpha=C.ALPHA_LOW)
        ssbc_90 = SSBCConformal(alpha=C.ALPHA_HIGH)
        q_95 = ssbc_95.calibrate(scores_cal)
        q_90 = ssbc_90.calibrate(scores_cal)

        # EnbPI rolling intervals on test
        logger.info("[BiVATScorer] >>> EnbPI rolling (lookback=24 mois)")
        enbpi_95 = EnbPI(alpha=C.ALPHA_LOW,  lookback=24)
        enbpi_90 = EnbPI(alpha=C.ALPHA_HIGH, lookback=24)

        test_valid = scores_test[~np.isnan(scores_test)]
        cal_valid  = scores_cal[~np.isnan(scores_cal)]

        q_roll_95, lo_95, hi_95 = enbpi_95.rolling_intervals(test_valid, cal_valid)
        q_roll_90, lo_90, hi_90 = enbpi_90.rolling_intervals(test_valid, cal_valid)

        # Anomaly counts at both levels
        anom_95 = ssbc_95.predict(scores_test)
        anom_90 = ssbc_90.predict(scores_test)
        n_valid_test = int((~np.isnan(scores_test)).sum())
        n_anom_95 = int(np.nansum(anom_95))
        n_anom_90 = int(np.nansum(anom_90))
        logger.info("[BiVATScorer] Anomalies détectées  95%%: %d/%d (%.1f%%)  90%%: %d/%d (%.1f%%)",
                    n_anom_95, n_valid_test, 100 * n_anom_95 / max(1, n_valid_test),
                    n_anom_90, n_valid_test, 100 * n_anom_90 / max(1, n_valid_test))
        logger.info("[BiVATScorer] q̂_95=%.5f · q̂_90=%.5f · q_roll_95∈[%.5f,%.5f] · q_roll_90∈[%.5f,%.5f]",
                    q_95, q_90,
                    float(np.nanmin(q_roll_95)), float(np.nanmax(q_roll_95)),
                    float(np.nanmin(q_roll_90)), float(np.nanmax(q_roll_90)))

        return {
            "scores_train": scores_train,
            "scores_cal":   scores_cal,
            "scores_test":  scores_test,
            "q_hat_95":     q_95,
            "q_hat_90":     q_90,
            "q_roll_95":    q_roll_95,
            "q_roll_90":    q_roll_90,
            "lo_95": lo_95, "hi_95": hi_95,
            "lo_90": lo_90, "hi_90": hi_90,
            "anomalies_95": anom_95,
            "anomalies_90": anom_90,
        }
