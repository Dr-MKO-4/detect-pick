"""
beac_lof/detection.py  Calcul du LOF* et du seuil τ (§5/6).

Implémentation entièrement PyTorch :
  - torch.cdist     : matrice de distances euclidiennes (n×n) en un seul appel
  - torch.sort      : k-ième voisin + indices des voisins par observation
  - torch.maximum   : reach-distance vectorisée (aucune boucle sur les paires)
  - torch.quantile  : seuil τ IQR et p95

Exporte :
  lof_star(X, n, lb, ub) → (df_lof_par_minpts, scores_lof_star)
  seuil_tau(scores)       → (tau_iqr, tau_p95)
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .config import MINPTS_LB, MINPTS_UB

log = logging.getLogger(__name__)


def _lof_k(D: torch.Tensor, k: int) -> torch.Tensor:
    """
    LOF_k pour tous les points, étant donné la matrice de distances D (n×n).

    Formules (Breunig et al. 2000) :
        k-dist(p)           = distance au k-ième plus proche voisin
        N_k(p)              = ensemble des k plus proches voisins de p
        reach-dist_k(p, o)  = max(k-dist(o), d(p, o))
        lrd_k(p)            = k / Σ_{o ∈ N_k(p)} reach-dist_k(p, o)
        LOF_k(p)            = mean_{o ∈ N_k(p)}[ lrd_k(o) / lrd_k(p) ]

    Paramètres
    ----------
    D : (n, n) matrice symétrique de distances (diagonale = 0)
    k : MinPts  nombre de voisins

    Retourne
    --------
    lof : (n,) scores LOF_k (float64)
    """
    n = D.shape[0]
    k = min(k, n - 1)

    # topk(k+1) : O(n·k) au lieu du full sort O(n·n·log n)
    # largest=False → k+1 plus petites distances (rang 0 = soi-même, dist ≈ 0)
    _topk_vals, idx_sorted = torch.topk(D, k + 1, dim=1, largest=False, sorted=True)
    k_dist  = _topk_vals[:, k]   # (n,)  k-ième voisin
    knn_idx = idx_sorted[:, 1:]  # (n, k)  skip soi-même (indice 0)

    # d(p, o) pour chaque o ∈ N_k(p)  indexation avancée
    row_idx  = torch.arange(n, device=D.device).unsqueeze(1).expand(n, k)
    knn_dist = D[row_idx, knn_idx]     # (n, k)

    # k-dist(o) pour chaque o ∈ N_k(p)
    k_dist_o = k_dist[knn_idx]         # (n, k)

    # reach-dist_k(p, o) = max(k-dist(o), d(p, o))  vectorisé
    reach = torch.maximum(k_dist_o, knn_dist)   # (n, k)

    # lrd_k(p) = k / Σ reach-dist  (clamp pour éviter la division par zéro)
    lrd = k / reach.sum(dim=1).clamp(min=1e-12)  # (n,)

    # LOF_k(p) = mean(lrd(o) / lrd(p)) sur N_k(p)
    lrd_o = lrd[knn_idx]               # (n, k)
    lof   = (lrd_o / lrd.unsqueeze(1).clamp(min=1e-12)).mean(dim=1)  # (n,)

    return lof


def lof_star(
    X: np.ndarray,
    n: int,
    lb: int = MINPTS_LB,
    ub: int = MINPTS_UB,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Calcule LOF(MinPts) pour chaque MinPts ∈ [lb, ub], puis agrège par le maximum
    (Breunig et al. 2000, §6.2) :

        LOF*(p) = max{ LOF_MinPts(p) | lb ≤ MinPts ≤ ub }

    Pour les fichiers avec n < 100 observations (RCA, Guinée équatoriale),
    la borne supérieure est ramenée à ⌊n/5⌋ (rapport MinPts/n cohérent).

    Toutes les opérations numériques sont vectorisées via PyTorch :
    - torch.cdist   : matrice de distances (n×n) en un seul appel BLAS
    - torch.sort    : k-ième voisin + indices des voisins
    - torch.maximum : reach-distance vectorisée
    - indexation avancée : lrd et LOF sans boucles sur les observations

    Retourne
    --------
    df_mp    : pd.DataFrame (n × len(range))  scores par MinPts (colonnes = MinPts int)
    lof_star : np.ndarray (n,)                score maximal par observation
    """
    if n < 100:
        ub_eff = max(lb, int(np.floor(n / 5)))
        log.warning(
            "n=%d < 100 : borne supérieure MinPts réduite de %d à %d "
            "(ratio MinPts/n cohérent). Les résultats ne sont pas directement "
            "comparables à ceux obtenus avec n ≥ 100.",
            n, ub, ub_eff,
        )
    else:
        ub_eff = ub
    minpts_range = list(range(lb, ub_eff + 1))
    n_minpts = len(minpts_range)

    log.info("    LOF*  n=%d obs, MinPts∈[%d,%d] (%d valeurs), X.shape=%s",
             n, lb, ub_eff, n_minpts, X.shape)
    t0 = time.time()

    X_t = torch.tensor(X, dtype=torch.float64)
    log.debug("    LOF* calcul matrice de distances cdist(%d×%d)…", n, n)
    t_cdist = time.time()
    D   = torch.cdist(X_t, X_t, p=2.0)          # (n, n)  matrice complète
    log.info("    LOF* matrice D(%d×%d) calculée en %.1fs", n, n, time.time() - t_cdist)

    scores_mat = torch.zeros(n, n_minpts, dtype=torch.float64)
    log_every  = max(1, n_minpts // 4)           # log ~4 fois pendant le balayage

    for j, mp in enumerate(minpts_range):
        scores_mat[:, j] = _lof_k(D, mp)
        if (j + 1) % log_every == 0 or (j + 1) == n_minpts:
            partial = scores_mat[:, : j + 1].max(dim=1).values.numpy()
            log.info("    LOF* MinPts=%d (%d/%d)  LOF* courant: μ=%.3f  max=%.3f  p95=%.3f",
                     mp, j + 1, n_minpts,
                     float(np.mean(partial)), float(np.max(partial)),
                     float(np.quantile(partial, 0.95)))

    lof_scores = scores_mat.max(dim=1).values    # (n,)  LOF*(p) = max sur MinPts
    lof_np     = lof_scores.numpy()
    df_mp      = pd.DataFrame(scores_mat.numpy(), columns=minpts_range)

    elapsed = time.time() - t0
    log.info("    LOF* terminé en %.1fs  μ=%.3f  σ=%.3f  max=%.3f  p75=%.3f  p95=%.3f",
             elapsed, float(np.mean(lof_np)), float(np.std(lof_np)),
             float(np.max(lof_np)), float(np.quantile(lof_np, 0.75)),
             float(np.quantile(lof_np, 0.95)))

    return df_mp, lof_np


def seuil_tau(scores: np.ndarray,
              method: str = "iqr15") -> Tuple[float, float]:
    """
    Seuil de détection (Tukey 1977 / Kriegel et al. 2011).

    method
    ------
    "iqr15" : τ = Q3 + 1.5 × IQR  (défaut)
    "iqr20" : τ = Q3 + 2.0 × IQR  (moins sensible)
    "p95"   : τ = percentile 95

    Retourne
    --------
    tau_principal : float  seuil selon la méthode choisie
    tau_p95       : float  seuil p95 (toujours calculé, variante)
    """
    t   = torch.tensor(scores, dtype=torch.float64)
    q   = torch.quantile(t, torch.tensor([0.25, 0.75, 0.95], dtype=torch.float64))
    q1, q3, p95 = q[0].item(), q[1].item(), q[2].item()
    iqr     = q3 - q1
    tau_iqr15 = q3 + 1.5 * iqr
    tau_iqr20 = q3 + 2.0 * iqr

    if method == "iqr20":
        tau_principal = tau_iqr20
    elif method == "p95":
        tau_principal = p95
    else:
        tau_principal = tau_iqr15

    n_above = int((scores > tau_principal).sum())
    n_above_p95 = int((scores > p95).sum())
    log.info("    Seuil τ [%s]  Q1=%.3f  Q3=%.3f  IQR=%.3f  τ=%.3f (%d anomalies)  p95=%.3f (%d anomalies)",
             method, q1, q3, iqr, tau_principal, n_above, p95, n_above_p95)

    return tau_principal, p95
