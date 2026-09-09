"""
beac_lof/preprocessing.py  Imputation et standardisation robuste (§2.2–2.3).

Implémentation entièrement PyTorch + Pandas :
  - torch.nanmedian    : médiane robuste vectorisée (colonne par colonne)
  - torch.linalg.lstsq : régression OLS itérée pour MICE (toutes les colonnes)
  - torch.nan_to_num   : initialisation des valeurs manquantes par la moyenne
  - Aucun recours à sklearn ou SciPy

Conformité methode.tex §2.3 :
  MICE utilise TOUTES les corrélations inter-séries (aucune restriction
  n_nearest_features)  contrairement à l'ancien IterativeImputer limité à 10.

Deux fonctions publiques :
  imputer(df)          → (df_imputed, excluded_cols)
  mad_standardize(df)  → df_standardized
"""

from __future__ import annotations

import logging
import time
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch

from .config import KAPPA, SHORT_GAP_MAX, INDICATOR_NAN_MAX

log = logging.getLogger(__name__)


# ── Standardisation MAD ────────────────────────────────────────────────────────

def mad_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardisation robuste médiane/MAD (Leys et al. 2013) :

        x* = (x − médiane(x)) / (κ · MAD(x)),   κ = 1.4826

    Point de rupture 50 % : les outliers n'influencent ni le centre ni l'échelle
    (Leys et al. 2013, §2).

    Si MAD = 0 (série quasi-constante), repli sur l'écart-type inter-colonne.

    Implémentation PyTorch :
    - torch.nanmedian  : médiane ignorant les NaN, vectorisée sur (n × p)
    - opérations de masquage et broadcasting sans boucles Python sur les colonnes
    """
    n, p = df.shape
    log.info("    MAD-standardisation : %d obs × %d indicateurs", n, p)

    arr      = torch.tensor(df.values, dtype=torch.float64)  # (n, p)
    nan_mask = arr.isnan()

    # ── Médiane par colonne (ignore NaN)
    med = torch.nanmedian(arr, dim=0).values  # (p,)

    # ── Déviations absolues  NaN préservés dans les positions originales
    dev = (arr - med).abs()
    dev[nan_mask] = float("nan")

    # ── MAD par colonne
    mad = torch.nanmedian(dev, dim=0).values  # (p,)

    # ── Repli écart-type là où MAD = 0
    zero_mad = mad == 0
    n_zero = int(zero_mad.sum().item())
    if n_zero > 0:
        zero_cols = [df.columns[i] for i in zero_mad.nonzero(as_tuple=True)[0].tolist()]
        log.info("    MAD=0 sur %d colonnes (repli écart-type) : %s%s",
                 n_zero, zero_cols[:5], " ..." if n_zero > 5 else "")
        arr_safe   = arr.nan_to_num(nan=0.0)
        valid_cnt  = (~nan_mask).sum(dim=0).double().clamp(min=1.0)  # (p,)
        col_mean   = arr_safe.sum(dim=0) / valid_cnt
        sq_diff    = (arr - col_mean).pow(2).nan_to_num(nan=0.0)
        std_vals   = (sq_diff.sum(dim=0) / (valid_cnt - 1).clamp(min=1.0)).sqrt()
        fallback   = std_vals.where(std_vals > 0, torch.ones_like(std_vals))
        denom      = torch.where(zero_mad, fallback, KAPPA * mad)
    else:
        denom = KAPPA * mad  # float × Tensor → Tensor, OK

    # ── Standardisation vectorisée
    result            = (arr - med) / denom.clamp(min=1e-12)
    result[nan_mask]  = float("nan")   # restaurer les NaN originaux

    med_np = med.numpy()
    mad_np = mad.numpy()
    log.info("    MAD-std terminé  médiane col. [min=%.3f, max=%.3f], MAD [min=%.4f, max=%.4f]",
             float(med_np.min()), float(med_np.max()),
             float(mad_np[mad_np > 0].min()) if (mad_np > 0).any() else 0.0,
             float(mad_np.max()))

    return pd.DataFrame(result.numpy(), index=df.index, columns=df.columns)


# ── MICE PyTorch ───────────────────────────────────────────────────────────────

def _mice_pytorch(
    arr: torch.Tensor,
    missing_mask: torch.Tensor,
    max_iter: int = 10,
) -> torch.Tensor:
    """
    MICE  Multiple Imputation by Chained Equations (Van Buuren 2011).

    Algorithme :
      Pour chaque itération, pour chaque colonne c avec des valeurs manquantes :
        1. Sélectionner les lignes observées (obs_mask) et manquantes (miss_mask)
        2. Régression OLS sur TOUTES les autres colonnes (p−1 prédicteurs)
           via torch.linalg.lstsq  aucune restriction sur le nombre de prédicteurs
           (conforme methode.tex §2.3 : utiliser toutes les corrélations inter-séries)
        3. Prédire et remplacer les valeurs manquantes de c

    Paramètres
    ----------
    arr          : (n, p) tenseur avec NaN remplacés par les moyennes colonnes
    missing_mask : (n, p) masque booléen  True là où la valeur est originellement manquante
    max_iter     : 10 itérations par défaut (conforme methode.tex §2.3)

    Retourne
    --------
    arr : (n, p) tenseur entièrement imputé
    """
    n, p = arr.shape
    cols_with_nan = missing_mask.any(dim=0).nonzero(as_tuple=True)[0].tolist()
    n_nan_cols = len(cols_with_nan)
    n_nan_cells = int(missing_mask.sum().item())
    _tol = 1e-4

    log.info("    MICE démarré  %d col. avec NaN (%d cellules manquantes), max_iter=%d, tol=%.0e",
             n_nan_cols, n_nan_cells, max_iter, _tol)

    # Masques et colonnes prédicteurs : indépendants des valeurs de `arr`,
    # donc constants d'une itération MICE à l'autre — précalculés une seule
    # fois au lieu de max_iter fois (jusqu'à 10x moins de recomputation).
    col_info = {}
    for c in cols_with_nan:
        obs_mask  = ~missing_mask[:, c]
        miss_mask =  missing_mask[:, c]
        if obs_mask.sum() < 2 or miss_mask.sum() == 0:
            continue
        pred_cols = [j for j in range(p) if j != c]
        col_info[c] = (obs_mask, miss_mask, pred_cols)

    converged_at = max_iter
    for _iter in range(max_iter):
        arr_prev = arr.clone()
        n_failed = 0
        for c, (obs_mask, miss_mask, pred_cols) in col_info.items():
            X_obs     = arr[obs_mask][:, pred_cols]    # (n_obs,  p-1)
            y_obs     = arr[obs_mask,  c]              # (n_obs,)
            X_miss    = arr[miss_mask][:, pred_cols]   # (n_miss, p-1)

            # Ajout de l'intercepte
            X_obs_i  = torch.cat([torch.ones(X_obs.shape[0],  1, dtype=arr.dtype), X_obs],  dim=1)
            X_miss_i = torch.cat([torch.ones(X_miss.shape[0], 1, dtype=arr.dtype), X_miss], dim=1)

            # OLS : β = (XᵀX)⁻¹Xᵀy via moindres carrés (gère les cas rang-déficient)
            try:
                sol   = torch.linalg.lstsq(X_obs_i, y_obs.unsqueeze(1), driver="gelsd")
                beta  = sol.solution           # (p, 1)
                preds = (X_miss_i @ beta).squeeze(1)
                arr[miss_mask, c] = preds
            except Exception:
                n_failed += 1   # conserver les valeurs courantes si la régression échoue

        # Arrêt précoce si les imputations ont convergé
        delta = (arr - arr_prev).abs().max().item()
        log.debug("    MICE iter %d/%d  Δmax=%.6f%s%s",
                  _iter + 1, max_iter, delta,
                  f" ({n_failed} OLS échecs)" if n_failed else "",
                  " [convergé]" if delta < _tol else "")
        if delta < _tol:
            converged_at = _iter + 1
            log.info("    MICE convergé à l'itération %d/%d (Δmax=%.2e < tol=%.0e)",
                     converged_at, max_iter, delta, _tol)
            break
    else:
        log.warning("    MICE non convergé après %d itérations (Δmax=%.4f)  résultats acceptables si Δ faible",
                    max_iter, delta)

    return arr


# ── Imputation en 3 régimes ────────────────────────────────────────────────────

def imputer(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List]:
    """
    Imputation en 3 régimes (methode.tex §2.3) :

    1. **Interpolation linéaire** (MAR court) :
       lacunes ≤ SHORT_GAP_MAX mois consécutifs.

    2. **Exclusion MNAR** :
       indicateurs dont la fraction imputée en tête de fenêtre dépasse
       INDICATOR_NAN_MAX (20 %)  biais d'imputation non contrôlable.

    3. **MICE PyTorch** (MAR étendu, Van Buuren & Groothuis-Oudshoorn 2011) :
       lacunes > SHORT_GAP_MAX mois, après exclusion MNAR.
       Régression OLS (torch.linalg.lstsq) itérée sur **toutes** les colonnes
       corrélées  aucune restriction n_nearest_features (conforme methode.tex).

    Retourne
    --------
    df_imp   : pd.DataFrame  données imputées (plus de NaN si possible)
    excluded : list          codes IFS des indicateurs exclus (MNAR ou tout-NaN)
    """
    df_imp  = df.copy()
    n_total = len(df_imp)
    n_cols_init = df_imp.shape[1]
    n_nan_init  = int(df_imp.isna().sum().sum())

    log.info("    Imputation démarrée  %d obs × %d ind. · %d cellules manquantes (%.1f%%)",
             n_total, n_cols_init, n_nan_init,
             100.0 * n_nan_init / max(1, n_total * n_cols_init))

    # ── Régime 1 : interpolation linéaire courte (Pandas, MAR ≤ SHORT_GAP_MAX)
    n_nan_before_interp = int(df_imp.isna().sum().sum())
    df_imp = df_imp.interpolate(method="time", limit=SHORT_GAP_MAX, limit_direction="both")
    n_filled_interp = n_nan_before_interp - int(df_imp.isna().sum().sum())
    log.info("    Régime 1 (interpolation linéaire ≤%d mois) : %d cellules comblées",
             SHORT_GAP_MAX, n_filled_interp)

    # ── Régime 2 : exclusion MNAR
    excluded: List = []
    leading_window = max(1, n_total // 5)
    for col in df_imp.columns:
        n_leading = df_imp[col].iloc[:leading_window].isna().sum()
        if n_leading / n_total > INDICATOR_NAN_MAX:
            excluded.append(col)
    if excluded:
        log.info("    Régime 2 (exclusion MNAR, seuil=%.0f%%) : %d indicateurs exclus  %s%s",
                 INDICATOR_NAN_MAX * 100, len(excluded),
                 excluded[:3], " ..." if len(excluded) > 3 else "")
        df_imp.drop(columns=excluded, inplace=True)

    # Colonnes entièrement NaN → exclusion avant MICE
    all_nan = df_imp.columns[df_imp.isna().all()].tolist()
    if all_nan:
        excluded.extend(all_nan)
        df_imp.drop(columns=all_nan, inplace=True)
        log.info("    Colonnes tout-NaN exclues supplémentaires : %d", len(all_nan))

    # ── Régime 3 : MICE PyTorch pour les lacunes résiduelles (> SHORT_GAP_MAX mois)
    n_nan_residuel = int(df_imp.isna().sum().sum())
    if n_nan_residuel > 0:
        log.info("    Régime 3 (MICE PyTorch) : %d cellules résiduelles à imputer sur %d ind.",
                 n_nan_residuel, df_imp.shape[1])
        t0 = time.time()
        arr_np  = df_imp.values.astype(float)
        miss_np = np.isnan(arr_np)

        arr_t  = torch.tensor(arr_np, dtype=torch.float64)
        miss_t = torch.tensor(miss_np, dtype=torch.bool)

        # Initialisation : valeurs manquantes ← moyenne de la colonne
        col_means = torch.nanmean(arr_t, dim=0)
        for j in range(arr_t.shape[1]):
            if miss_t[:, j].any():
                arr_t[miss_t[:, j], j] = col_means[j]

        arr_t  = _mice_pytorch(arr_t, miss_t, max_iter=10)
        df_imp = pd.DataFrame(arr_t.numpy(), index=df_imp.index, columns=df_imp.columns)
        log.info("    MICE terminé en %.1fs  NaN restants : %d",
                 time.time() - t0, int(df_imp.isna().sum().sum()))
    else:
        log.info("    Régime 3 (MICE) ignoré  aucune lacune résiduelle après interpolation")

    log.info("    Imputation terminée  %d ind. retenus (%d exclus)",
             df_imp.shape[1], len(excluded))
    return df_imp, excluded
