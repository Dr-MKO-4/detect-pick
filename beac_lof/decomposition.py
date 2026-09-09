"""
beac_lof/decomposition.py  Décomposition STL des séries temporelles (§3).

Exporte :
  stl_decompose(df) → (residus, composantes)
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .config import STL_PERIOD

log = logging.getLogger(__name__)


def _stl_one(args):
    """
    Worker sérialisable pour joblib.Parallel.

    Returns
    -------
    (col, obs, trend, seasonal, resid, fallback_reason)
      fallback_reason: "" si STL réussi, sinon la raison du repli rolling-mean.
    """
    col, values, index, period = args
    serie = pd.Series(values, index=index, dtype=float)

    if serie.isna().any():
        trend_v = serie.rolling(period, center=True, min_periods=1).mean()
        return col, serie.values, trend_v.values, np.zeros(len(serie)), (serie - trend_v).values, "NaN_residuels"

    if len(serie) < 2 * period:
        trend_v = serie.rolling(period, center=True, min_periods=1).mean()
        return col, serie.values, trend_v.values, np.zeros(len(serie)), (serie - trend_v).values, f"serie_trop_courte({len(serie)}<{2*period})"

    try:
        res = STL(serie, period=period, robust=True).fit()
        return col, serie.values, res.trend, res.seasonal, res.resid, ""
    except Exception as exc:
        trend_v = serie.rolling(period, center=True, min_periods=1).mean()
        return col, serie.values, trend_v.values, np.zeros(len(serie)), (serie - trend_v).values, f"STL_exception({exc})"


def stl_decompose(df: pd.DataFrame,
                  period: int | None = None) -> Tuple[pd.DataFrame, dict]:
    """
    Décompose chaque colonne de `df` selon le modèle additif STL
    (Cleveland et al. 1990, période = STL_PERIOD = 12) :

        x_{i,t} = T_{i,t} + S_{i,t} + R_{i,t}

    `robust=True` : pondération interne qui atténue l'influence des anomalies
    sur l'estimation de la tendance, évitant que les points recherchés biaisent
    la décomposition elle-même.

    Pour les séries trop courtes (< 2 périodes) ou avec NaN résiduels,
    la tendance est estimée par rolling-mean et la saisonnalité est posée à 0.

    Retourne
    --------
    residus     : pd.DataFrame  R_{i,t} (n_mois × n_indicateurs)
    composantes : dict["observed"|"trend"|"seasonal"|"residual" → pd.DataFrame]
    """
    _period = period if period is not None else STL_PERIOD
    n_cols = len(df.columns)
    log.info("    STL décomposition  %d séries (période=%d, robust=True)", n_cols, _period)
    t0 = time.time()

    # En dessous de ce seuil, le coût de démarrage du pool de process
    # (sérialisation + un aller-retour IPC par colonne) dépasse le gain de
    # parallélisation — on reste séquentiel.
    _MIN_COLS_FOR_PARALLEL = 8

    try:
        from joblib import Parallel, delayed
        if n_cols >= _MIN_COLS_FOR_PARALLEL:
            n_jobs = -1
            backend_label = "joblib/loky parallèle"
        else:
            Parallel = None
            n_jobs = 1
            backend_label = f"séquentiel ({n_cols} série(s) < seuil {_MIN_COLS_FOR_PARALLEL})"
    except ImportError:
        Parallel = None
        n_jobs   = 1
        backend_label = "séquentiel (joblib absent)"

    log.info("    STL backend : %s", backend_label)
    args_list = [(col, df[col].astype(float).values, df.index, _period) for col in df.columns]

    if Parallel is not None:
        raw = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_stl_one)(a) for a in args_list
        )
    else:
        raw = [_stl_one(a) for a in args_list]

    observed = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    trend    = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    seasonal = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    residual = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    n_ok = 0
    fallbacks: list[tuple[str, str]] = []
    for col, obs_v, trend_v, seas_v, resid_v, fallback_reason in raw:
        observed[col] = obs_v
        trend[col]    = trend_v
        seasonal[col] = seas_v
        residual[col] = resid_v
        if fallback_reason:
            fallbacks.append((col, fallback_reason))
        else:
            n_ok += 1

    elapsed = time.time() - t0
    log.info("    STL terminé en %.1fs  %d/%d STL complets, %d replis rolling-mean",
             elapsed, n_ok, n_cols, len(fallbacks))
    if fallbacks:
        for col, reason in fallbacks[:5]:
            log.warning("    STL repli [%s] : %s", col, reason)
        if len(fallbacks) > 5:
            log.warning("    ... et %d autres replis (voir logs DEBUG pour le détail)", len(fallbacks) - 5)

    # Statistiques globales des résidus
    resid_vals = residual.values.astype(float)
    finite_mask = np.isfinite(resid_vals)
    if finite_mask.any():
        log.info("    Résidus STL  μ=%.4f  σ=%.4f  |max|=%.4f",
                 float(np.nanmean(resid_vals)),
                 float(np.nanstd(resid_vals)),
                 float(np.nanmax(np.abs(resid_vals))))

    composantes = {
        "observed": observed.astype(float),
        "trend":    trend.astype(float),
        "seasonal": seasonal.astype(float),
        "residual": residual.astype(float),
    }
    return residual.astype(float), composantes
