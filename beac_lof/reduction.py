"""
beac_lof/reduction.py  RPCA et ACP sur la composante de bas rang (§4).

Implémentation entièrement PyTorch :
  - torch.linalg.svd  : SVD économique pour RPCA-IALM et ACP
  - torch.linalg.norm : normes de Frobenius et spectrale
  - seuillage doux vectorisé (aucune boucle numpy/sklearn)

Exporte :
  rpca_ialm(X)              → (L, S)
  pca_sur_L(L, indicateurs) → (projections, loadings_df, variance_ratio)
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .config import PCA_VAR_TARGET

log = logging.getLogger(__name__)


# ── RPCA ──────────────────────────────────────────────────────────────────────

def rpca_ialm(
    X: np.ndarray,
    lam: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Décomposition Robust PCA : X ≈ L + S (Candès et al. 2011).

    Résout le problème convexe :
        min_{L,S}  ||L||_*  +  λ ||S||_1     s.c.  L + S = X

    par la méthode Inexact ALM (Lin et al. 2010) :
    - Seuillage doux des valeurs singulières pour L  (proximal de ||·||_*)
    - Seuillage doux par élément pour S              (proximal de ||·||_1)
    - Mise à jour de la variable duale Y et du paramètre de pénalité μ

    λ = 1 / √max(m, n) par défaut (Candès et al. 2011, Théorème 1.1).

    Implémentation PyTorch :
    - torch.linalg.svd    : SVD économique à chaque itération
    - seuillage vectorisé : (sigma - 1/μ).clamp(min=0) et T.sign() * (|T| - λ/μ).clamp(min=0)
    - torch.linalg.norm   : normes Frobenius et spectrale

    Retourne
    --------
    L : np.ndarray  composante de bas rang (structure normale)
    S : np.ndarray  composante creuse (anomalies ponctuelles)
    """
    m, n = X.shape
    lam = lam if lam is not None else (1.0 / (max(m, n) ** 0.5))

    log.info("    RPCA-IALM démarré  X=(%d×%d), λ=%.4f, tol=%.0e, max_iter=%d",
             m, n, lam, tol, max_iter)
    t0 = time.time()

    Xt = torch.tensor(X, dtype=torch.float64)

    norm_fro  = torch.linalg.norm(Xt, ord="fro").item()
    norm_two  = torch.linalg.norm(Xt, ord=2).item()
    norm_inf  = Xt.abs().max().item() / lam
    dual_norm = max(norm_two, norm_inf)

    Y      = Xt / dual_norm
    mu     = 1.25 / norm_two
    mu_max = mu * 1e7
    rho    = 1.5
    L      = torch.zeros_like(Xt)
    S      = torch.zeros_like(Xt)

    converged_at = max_iter
    last_residual = float("nan")
    for _iter in range(max_iter):
        # Prox norme nucléaire : SVD + seuillage doux des valeurs singulières
        U, sigma, Vh = torch.linalg.svd(Xt - S + Y / mu, full_matrices=False)
        sigma_t = (sigma - 1.0 / mu).clamp(min=0.0)
        L_new   = (U * sigma_t) @ Vh          # (m,r)·diag(σ_t)·(r,n) sans matrice diag

        # Prox norme ℓ₁ : seuillage doux par élément
        T     = Xt - L_new + Y / mu
        S_new = T.sign() * (T.abs() - lam / mu).clamp(min=0.0)

        Z        = Xt - L_new - S_new
        residual = torch.linalg.norm(Z, ord="fro").item() / (norm_fro + 1e-12)
        Y        = Y + mu * Z
        mu       = min(rho * mu, mu_max)
        L, S     = L_new, S_new
        last_residual = residual

        # Log tous les 50 pas ou à la convergence
        if (_iter + 1) % 50 == 0:
            rank_L   = int((sigma_t > 1e-10).sum().item())
            sparsity = float((S.abs() > 1e-4).float().mean().item()) * 100
            log.info("    RPCA iter %4d  résidu=%.2e  rang(L)=%d  densité(S)=%.1f%%",
                     _iter + 1, residual, rank_L, sparsity)

        if residual < tol:
            converged_at = _iter + 1
            break

    elapsed = time.time() - t0
    rank_L_final   = int(np.linalg.matrix_rank(L.numpy()))
    sparsity_final = float((S.abs() > 1e-4).float().mean().item()) * 100
    if converged_at < max_iter:
        log.info("    RPCA convergé à l'itération %d/%d (résidu=%.2e) en %.1fs  rang(L)=%d, densité(S)=%.1f%%",
                 converged_at, max_iter, last_residual, elapsed, rank_L_final, sparsity_final)
    else:
        log.warning("    RPCA non convergé après %d itérations (résidu=%.2e, tol=%.0e) en %.1fs  rang(L)=%d, densité(S)=%.1f%%",
                    max_iter, last_residual, tol, elapsed, rank_L_final, sparsity_final)

    return L.numpy(), S.numpy()


# ── ACP sur L ─────────────────────────────────────────────────────────────────

def _select_n_components(variance: np.ndarray, target: float = PCA_VAR_TARGET) -> int:
    """Nombre minimal de composantes pour atteindre `target` de variance cumulée."""
    cumvar     = np.cumsum(variance)
    candidates = np.where(cumvar >= target)[0]
    if len(candidates) == 0:
        log.warning(
            "ACP : cible de variance %.0f%% non atteinte (max = %.1f%%)  "
            "toutes les composantes retenues.",
            target * 100, cumvar[-1] * 100,
        )
        return len(variance)
    return int(candidates[0]) + 1


def pca_sur_L(
    L: np.ndarray,
    indicateurs: pd.Index,
    var_target: float = PCA_VAR_TARGET,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    ACP sur la composante de bas rang L via SVD (Zimek et al. 2012 : nécessaire
    car p > 20 dégénère les distances euclidiennes du LOF).

    Choisit automatiquement k = min composantes pour atteindre `var_target`
    de variance cumulée.

    Implémentation PyTorch (torch.linalg.svd) :
      1. Centrage          : L_c = L − mean(L, axis=0)
      2. SVD économique    : L_c = U Σ Vᵀ
      3. Variance par composante : σᵢ² / Σ σⱼ²  (proportion de variance)
      4. Projections       : U_k · Σ_k  ≡  L_c @ V_k
      5. Loadings          : premières k lignes de Vᵀ (vecteurs propres)

    Retourne
    --------
    projections : np.ndarray  (n_obs × k)  coordonnées dans l'espace réduit
    loadings_df : pd.DataFrame (k × p)    vecteurs propres (lignes = composantes)
    var_ratio   : np.ndarray  (r,)        variance expliquée par composante
    """
    Lt  = torch.tensor(L, dtype=torch.float64)
    n, p = Lt.shape

    log.info("    ACP sur L  matrice (%d×%d), cible variance=%.0f%%", n, p, var_target * 100)

    # Centrage
    Lt_c = Lt - Lt.mean(dim=0, keepdim=True)

    # SVD économique : Lt_c = U Σ Vᵀ,  formes : U(n,r), sigma(r,), Vh(r,p)
    U, sigma, Vh = torch.linalg.svd(Lt_c, full_matrices=False)

    # Proportion de variance expliquée par composante
    var_sq    = sigma.pow(2)
    var_ratio = (var_sq / var_sq.sum().clamp(min=1e-12)).numpy()  # (r,)

    # Sélection des k premières composantes
    k = _select_n_components(var_ratio, var_target)

    cumvar_k = float(var_ratio[:k].sum()) * 100
    log.info("    ACP : %d composantes sélectionnées (variance cumulée = %.1f%% sur %.0f%% cible)",
             k, cumvar_k, var_target * 100)
    # Détail par composante retenue
    for i in range(min(k, 5)):
        log.info("      CP%d : %.2f%% de variance (σ=%.4f)", i + 1, var_ratio[i] * 100, sigma[i].item())
    if k > 5:
        log.info("      ... CP6–CP%d : %.2f%% de variance cumulée", k, (cumvar_k - var_ratio[:5].sum() * 100))

    # Projections (n × k) : U_k · Σ_k  ≡  L_c @ V_k
    projections = (U[:, :k] * sigma[:k]).numpy()

    # Loadings (k × p) : premières k lignes de Vᵀ
    loadings_df = pd.DataFrame(Vh[:k].numpy(), columns=indicateurs)

    return projections, loadings_df, var_ratio
