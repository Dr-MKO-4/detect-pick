"""BiVAT training loop  ELBO + Association Discrepancy minimax (SPEC §4.2 D4).

Loss:
    L = E[||X - X̂||²]                         ← MSE reconstruction
      - β · KL(q(z|X) || p(z))                  ← VAE KL regularisation
      + λ_ad · AssocDiscrepancy(Attn, Prior)     ← association discrepancy

Minimax training (Xu et al. 2022):
    Phase 1 (minimize) : minimize ELBO + λ_ad · AD  → pushes anomalies away
    Phase 2 (maximize) : maximize λ_ad · AD alone   → amplifies contrast
"""
from __future__ import annotations

import os
import logging
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from bivat.architecture import BiVAT
from bivat import config as C
from bivat.utils import assoc_discrepancy_mean as _assoc_discrepancy, assoc_discrepancy
from utils.hardware import DEVICE, AMP_ENABLED

logger = logging.getLogger(__name__)


# ── Loss components ───────────────────────────────────────────────────────────

def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(μ,σ²) || N(0,I))  Kingma & Welling (2013)."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()


def _elbo(x: torch.Tensor, out: dict, beta: float = C.BETA) -> torch.Tensor:
    """Perte à MINIMISER = -ELBO = reconstruction MSE + β · KL.

    KL(q(z|x) || p(z)) est toujours ≥ 0 (cf. _kl_divergence). La SOUSTRAIRE
    (mse - beta*kl) incite l'optimiseur à maximiser le KL sans limite —
    exactement le comportement observé en pratique (KL explosant de 0,1 à
    10^25 en 20 époques). Le terme doit être ADDITIONNÉ, comme dans la
    formulation standard de la loss VAE (Kingma & Welling, 2013)."""
    mse = F.mse_loss(out["x_hat"], x)
    kl  = _kl_divergence(out["mu"], out["logvar"])
    return mse + beta * kl


# ── Window dataset ────────────────────────────────────────────────────────────

def make_windows(X: np.ndarray, window: int = C.WINDOW_SIZE) -> torch.Tensor:
    """Slide window over T axis → (n_windows, window, d)."""
    T, d = X.shape
    if T < window:
        raise ValueError(f"Series length {T} < window {window}")
    slices = [X[i: i + window] for i in range(T - window + 1)]
    return torch.tensor(np.stack(slices), dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model: BiVAT,
    X_train: np.ndarray,
    epochs: int = C.EPOCHS,
    batch_size: int = C.BATCH_SIZE,
    lr: float = C.LR,
    lambda_ad: float = C.LAMBDA_AD,
    window: int = C.WINDOW_SIZE,
    beta: float = C.BETA,
    log_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """
    Train BiVAT on X_train (T × d array, already normalised).

    Returns
    -------
    history : list of {epoch, loss, mse, kl, ad}
    """
    def _log(msg: str, level: str = "info"):
        getattr(logger, level)(msg)
        if log_callback:
            log_callback(msg)

    import time as _time
    windows = make_windows(X_train, window=window).to(DEVICE)
    dataset = TensorDataset(windows)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    n_windows  = len(windows)
    n_batches_total = len(loader)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler    = torch.cuda.amp.GradScaler() if AMP_ENABLED else None

    _log(f"[BIVAT] Début entraînement  {epochs} époques · {n_windows} fenêtres "
         f"· batch={batch_size} · {n_batches_total} batches/époque · device={DEVICE}")
    _log(f"[BIVAT] Hyperparamètres  lr={lr:.0e}  β={beta}  λ_AD={lambda_ad}  "
         f"patience={10}  window={window}  d_in={X_train.shape[1]}")

    history = []
    _best_loss    = float("inf")
    _patience_cnt = 0
    _PATIENCE     = 10
    t_train_start = _time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_mse = epoch_kl = epoch_ad = 0.0
        n_batches = 0
        t_epoch   = _time.time()

        for (xb,) in loader:
            xb = xb.to(DEVICE)

            # ── Phase 1 : minimize ELBO + λ_ad·AD ──
            optimizer.zero_grad()
            if AMP_ENABLED:
                with torch.cuda.amp.autocast():
                    out    = model(xb)
                    mse    = F.mse_loss(out["x_hat"], xb)
                    kl     = _kl_divergence(out["mu"], out["logvar"])
                    ad     = _assoc_discrepancy(out["attn"])
                    loss_1 = mse + beta * kl + lambda_ad * ad
                scaler.scale(loss_1).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out    = model(xb)
                mse    = F.mse_loss(out["x_hat"], xb)
                kl     = _kl_divergence(out["mu"], out["logvar"])
                ad     = _assoc_discrepancy(out["attn"])
                loss_1 = mse + beta * kl + lambda_ad * ad
                loss_1.backward()
                optimizer.step()

            # ── Phase 2 : maximize λ_ad·AD  encoder-only, skip VAE + decoder ──
            optimizer.zero_grad()
            if AMP_ENABLED:
                with torch.cuda.amp.autocast():
                    attn2  = model.forward_encoder_only(xb)
                    ad2    = _assoc_discrepancy(attn2)
                    loss_2 = -lambda_ad * ad2
                scaler.scale(loss_2).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                attn2  = model.forward_encoder_only(xb)
                ad2    = _assoc_discrepancy(attn2)
                loss_2 = -lambda_ad * ad2
                loss_2.backward()
                optimizer.step()

            epoch_mse += mse.item()
            epoch_kl  += kl.item()
            epoch_ad  += ad.item()
            n_batches += 1

        avg_mse  = epoch_mse / n_batches
        avg_kl   = epoch_kl  / n_batches
        avg_ad   = epoch_ad  / n_batches
        avg_loss = avg_mse + beta * avg_kl + lambda_ad * avg_ad
        epoch_time = _time.time() - t_epoch

        avg = {
            "epoch": epoch,
            "loss":  avg_loss,
            "mse":   avg_mse,
            "kl":    avg_kl,
            "ad":    avg_ad,
            "time_s": epoch_time,
        }
        history.append(avg)

        # Early stopping check
        improved = avg_loss < _best_loss - 1e-4
        if improved:
            _best_loss    = avg_loss
            _patience_cnt = 0
        else:
            _patience_cnt += 1

        # Log chaque époque (concis) ou en détail si premier/dernier/jalons
        is_milestone = (epoch % 10 == 0 or epoch == 1 or epoch == epochs)
        if is_milestone:
            patience_str = f"  patience={_patience_cnt}/{_PATIENCE}" if _patience_cnt > 0 else ""
            _log(f"[BIVAT] Époque {epoch:4d}/{epochs}  loss={avg_loss:.5f}  "
                 f"MSE={avg_mse:.5f}  KL={avg_kl:.4f}  AD={avg_ad:.4f}  "
                 f"({epoch_time:.1f}s/ép){patience_str}")
        else:
            logger.debug("[BIVAT] Époque %4d  loss=%.5f  MSE=%.5f  KL=%.4f  AD=%.4f",
                         epoch, avg_loss, avg_mse, avg_kl, avg_ad)

        if _patience_cnt >= _PATIENCE:
            msg = (f"[BIVAT] Early stopping à l'époque {epoch}/{epochs} "
                   f" perte stagnante depuis {_PATIENCE} époques (meilleure perte={_best_loss:.5f})")
            _log(msg)
            break

    elapsed_total = _time.time() - t_train_start
    last = history[-1]
    _log(f"[BIVAT] Entraînement terminé  {last['epoch']} époques en {elapsed_total:.1f}s "
         f"| loss finale={last['loss']:.5f}  MSE={last['mse']:.5f}  KL={last['kl']:.4f}  AD={last['ad']:.4f}")
    return history


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model: BiVAT, path: str = C.WEIGHTS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "d_in": model.d_in,
        "d_model": model.d_model,
    }, path)
    logger.info(f"[BIVAT] Poids sauvegardés → {path}")


def load_checkpoint(path: str = C.WEIGHTS_PATH) -> BiVAT:
    ckpt  = torch.load(path, map_location=DEVICE)
    model = BiVAT(d_in=ckpt["d_in"], d_model=ckpt.get("d_model", C.D_MODEL))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    logger.info(f"[BIVAT] Poids chargés depuis {path}")
    return model
