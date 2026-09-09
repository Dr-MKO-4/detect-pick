"""bivat/utils.py  Utilitaires partagés entre training.py et scoring.py."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def assoc_discrepancy(attn: torch.Tensor) -> torch.Tensor:
    """
    Association Discrepancy (Xu et al. 2022, §3.2).

    Symétric KL divergence entre la distribution d'attention A et la prior
    gaussienne Q centrée sur chaque token.

        AD(A, Q) = 0.5 * [KL(A‖Q) + KL(Q‖A)]

    Paramètres
    ----------
    attn : (B, T, T)  poids d'attention softmax
    Retourne
    --------
    ad   : (B,)  score AD par échantillon
    """
    B, T, _ = attn.shape
    idx   = torch.arange(T, device=attn.device, dtype=attn.dtype)
    sigma = max(T / 4.0, 1.0)
    dist  = (idx.unsqueeze(0) - idx.unsqueeze(1)).pow(2) / (2 * sigma ** 2)
    Q     = F.softmax(-dist, dim=-1).unsqueeze(0).expand(B, -1, -1)
    eps   = 1e-8
    kl_aq = (attn * (attn + eps).log() - attn * (Q + eps).log()).sum(dim=-1).mean(dim=-1)
    kl_qa = (Q * (Q + eps).log() - Q * (attn + eps).log()).sum(dim=-1).mean(dim=-1)
    return 0.5 * (kl_aq + kl_qa)


def assoc_discrepancy_mean(attn: torch.Tensor) -> torch.Tensor:
    """Retourne la moyenne sur le batch → scalar."""
    return assoc_discrepancy(attn).mean()
