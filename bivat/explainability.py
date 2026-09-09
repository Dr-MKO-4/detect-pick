"""BiVAT explainability  SHAP GradientExplainer (SPEC §4.1 D3).

Computes DeepSHAP attributions φ_d(x_t) per indicator d at anomalous month t.
Complexity: O(d·T) = O(55 × 195) ≈ 10 725  tractable on CPU.

Usage
-----
exp = BiVATExplainer(model, X_background)
shap_values = exp.explain(X_anomalies)   # (T_anom, T_window, d)
top15 = exp.top_indicators(shap_values, n=15)
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from bivat.architecture import BiVAT
from bivat.training import make_windows
from bivat import config as C
from utils.hardware import DEVICE

logger = logging.getLogger(__name__)


class _BiVATWrapper(torch.nn.Module):
    """Thin wrapper exposing reconstruction score for SHAP."""
    def __init__(self, model: BiVAT):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # Per-sample MSE reconstruction error, shape (B, 1) — shap.GradientExplainer
        # indexe la sortie comme (B, n_outputs) ; un vecteur (B,) plat fait
        # planter cette indexation ("too many indices for tensor of dimension 1").
        mse = torch.nn.functional.mse_loss(out["x_hat"], x, reduction="none").mean(dim=[1, 2])
        return mse.unsqueeze(-1)


class BiVATExplainer:
    """
    SHAP GradientExplainer wrapper for BiVAT.

    Parameters
    ----------
    model       : trained BiVAT
    X_background: background set (n_bg × window × d), used as reference by SHAP
    """

    def __init__(self, model: BiVAT, X_background: np.ndarray):
        try:
            import shap as _shap
        except ImportError:
            raise ImportError("shap is required: pip install shap")

        model.eval()
        model.to(DEVICE)
        self.model   = model
        self.wrapper = _BiVATWrapper(model).to(DEVICE)

        bg = torch.tensor(X_background, dtype=torch.float32).to(DEVICE)
        # GradientExplainer plutôt que DeepExplainer : le backend PyTorch de
        # DeepExplainer s'appuie sur des hooks d'autograd qui ne suivent plus
        # les versions récentes de PyTorch pour une architecture avec
        # Transformer/BiLSTM (échec observé : "tuple index out of range" dès
        # l'initialisation). GradientExplainer (expected gradients) reste
        # compatible et convient à ce type d'architecture.
        self.explainer = _shap.GradientExplainer(self.wrapper, bg)

    def explain(self, X_windows: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for each window in X_windows.

        Parameters
        ----------
        X_windows : (n_anom, window, d)

        Returns
        -------
        shap_vals : (n_anom, window, d)   signed attributions
        """
        import time as _time
        logger.info("[SHAP] GradientExplainer.shap_values  %d fenêtres anormales (%d×%d×%d)",
                    X_windows.shape[0], *X_windows.shape)
        t0 = _time.time()
        t = torch.tensor(X_windows, dtype=torch.float32).to(DEVICE)
        sv = self.explainer.shap_values(t)
        # shap_values may be a list of arrays for multi-output  take index 0
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv)
        # La sortie du wrapper est (B, 1) → GradientExplainer ajoute une
        # dimension finale de taille 1 (n_anom, window, d, 1) : la retirer
        # pour retrouver la forme attendue (n_anom, window, d).
        if sv.ndim == 4 and sv.shape[-1] == 1:
            sv = sv[..., 0]
        logger.info("[SHAP] shap_values calculées en %.1fs  shape=%s  |φ|_max=%.4f  |φ|_moy=%.5f",
                    _time.time() - t0, sv.shape,
                    float(np.abs(sv).max()), float(np.abs(sv).mean()))
        return sv

    @staticmethod
    def aggregate_over_window(shap_vals: np.ndarray) -> np.ndarray:
        """
        Aggregate shap_vals (n_anom, window, d) over the window dimension
        (mean absolute) → (n_anom, d).
        """
        return np.abs(shap_vals).mean(axis=1)

    @staticmethod
    def top_indicators(
        shap_agg: np.ndarray,     # (n_anom, d)
        feature_names: list[str],
        n: int = C.SHAP_TOP_N,
    ) -> list[dict]:
        """
        For each anomalous month, return the top-n indicators sorted by |φ|.

        Returns
        -------
        list of length n_anom, each element is a dict:
            {
              "rank":   [1, 2, …, n]
              "name":   [indicator_code, …]
              "shap":   [signed_value, …]   # raw (not abs)
            }
        """
        results = []
        for row in shap_agg:
            order = np.argsort(np.abs(row))[::-1][:n]
            results.append({
                "rank":  list(range(1, len(order) + 1)),
                "name":  [feature_names[i] for i in order],
                "shap":  [float(row[i]) for i in order],
            })
        return results


# ── Convenience function ──────────────────────────────────────────────────────

def explain_top_anomalies(
    model: BiVAT,
    X: np.ndarray,
    scores: np.ndarray,
    feature_names: list[str],
    n_top: int = C.SHAP_TOP_N,
    n_bg:  int = C.SHAP_N_BG,
    window: int = C.WINDOW_SIZE,
) -> dict:
    """
    Full pipeline: find top-n_top anomalies, build explainer, compute SHAP.

    Parameters
    ----------
    X            : full normalised data (T × d)
    scores       : s_discrim (T,)  NaN allowed
    feature_names: list of d indicator names

    Returns
    -------
    dict with keys:
        top_indices   : list of n_top time indices (last step of each window)
        shap_values   : (n_top, window, d)
        top_indicators: list of dicts from BiVATExplainer.top_indicators()
    """
    import time as _time
    t0_total = _time.time()

    # Find top anomaly windows
    valid_mask = ~np.isnan(scores)
    n_valid    = int(valid_mask.sum())
    valid_idx  = np.where(valid_mask)[0]
    order      = valid_idx[np.argsort(scores[valid_idx])[::-1]]
    top_t      = order[:n_top].tolist()

    logger.info("[SHAP] explain_top_anomalies  %d anomalies à expliquer sur %d scores valides",
                len(top_t), n_valid)
    logger.info("[SHAP] Top anomalies  indices temporels : %s  scores : %s",
                top_t[:5],
                [round(float(scores[t]), 4) for t in top_t[:5]])

    windows_all = make_windows(X, window)   # (n_windows, window, d)
    n_windows   = len(windows_all)
    logger.info("[SHAP] %d fenêtres totales (%d×%d×%d)", n_windows, n_windows, window, X.shape[1])

    # Window index i → last time step = i + window - 1
    # Reverse: time step t → window index = t - window + 1
    top_win_idx  = [t - window + 1 for t in top_t if t - window + 1 >= 0]
    X_anomalies  = windows_all[top_win_idx].numpy()
    logger.info("[SHAP] %d fenêtres anormales extraites (sur %d demandées)",
                len(top_win_idx), n_top)

    # Background set: random sample of normal windows
    score_p75 = np.nanquantile(scores, 0.75)
    normal_win_idx = [
        i for i in range(len(windows_all))
        if not valid_mask[i + window - 1] or scores[i + window - 1] <= score_p75
    ]
    rng    = np.random.default_rng(42)
    n_bg_eff = min(n_bg, len(normal_win_idx))
    bg_idx = rng.choice(normal_win_idx, size=n_bg_eff, replace=False)
    X_bg   = windows_all[bg_idx].numpy()
    logger.info("[SHAP] Jeu de référence (background) : %d fenêtres normales (p75 score = %.4f)",
                n_bg_eff, float(score_p75))

    logger.info("[SHAP] Initialisation GradientExplainer…")
    t0_expl = _time.time()
    explainer  = BiVATExplainer(model, X_bg)
    logger.info("[SHAP] GradientExplainer initialisé en %.1fs", _time.time() - t0_expl)

    shap_vals  = explainer.explain(X_anomalies)              # (n_top, window, d)
    shap_agg   = BiVATExplainer.aggregate_over_window(shap_vals)  # (n_top, d)
    top_ind    = BiVATExplainer.top_indicators(shap_agg, feature_names)

    # Also compute signed mean over window (for directional bar chart)
    shap_signed = shap_vals.mean(axis=1)                     # (n_top, d)

    # Log top contributing indicators across all explained anomalies
    mean_abs_phi = np.abs(shap_agg).mean(axis=0)             # (d,)
    top_global   = np.argsort(mean_abs_phi)[::-1][:5]
    logger.info("[SHAP] Top-5 indicateurs (|φ| moyen sur toutes anomalies) : %s",
                [(feature_names[i], round(float(mean_abs_phi[i]), 4)) for i in top_global])
    logger.info("[SHAP] explain_top_anomalies terminé en %.1fs total", _time.time() - t0_total)

    return {
        "top_t":          top_t[:len(top_win_idx)],
        "shap_values":    shap_vals,
        "shap_agg":       shap_agg,
        "shap_signed":    shap_signed,
        "top_indicators": top_ind,
        "feature_names":  feature_names,
    }
