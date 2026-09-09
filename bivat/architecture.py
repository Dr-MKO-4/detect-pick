"""BiVAT architecture  Bidirectional Transformer-VAE + BiLSTM (SPEC §4.1 D1).

Architecture:
    X (T×d_in)
    → Linear projection → d_model
    → Sinusoidal positional encoding
    → Bi-Transformer encoder (L=2, H=4 heads, d_model=64)  captures attention A_t
    → BiLSTM (hidden=64, 2 layers)
    → Linear heads → μ_z, log σ²_z  (latent_dim=16)
    → Reparameterization trick → z
    → Decoder Transformer (L=2, d_model=64)
    → Linear projection → X̂ (T×d_in)

The forward pass also returns:
    - attention matrices A (for Association Discrepancy)
    - μ_z, log σ²_z (for KL divergence)
    - X̂ (for reconstruction)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from bivat import config as C


# ── Positional encoding (Vaswani 2017) ────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T > self.pe.size(1):
            raise ValueError(
                f"SinusoidalPE : T={T} dépasse max_len={self.pe.size(1)}. "
                f"Instanciez BiVAT avec max_len>{T}."
            )
        return x + self.pe[:, :T]


# ── Bidirectional Transformer encoder ─────────────────────────────────────────
# PyTorch TransformerEncoder is unidirectional by default.
# True bidirectionality is achieved by processing the sequence in both
# directions and concatenating, then projecting back to d_model.

class _BiTransformerLayer(nn.Module):
    """One bidirectional encoder layer = forward + backward TransformerEncoderLayer."""
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.fwd = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.bwd = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_f = self.fwd(x)                         # (B, T, d)
        h_b = self.bwd(x.flip(1)).flip(1)         # (B, T, d)
        h = self.proj(torch.cat([h_f, h_b], dim=-1))
        # Approximate attention: mean of last-head attention weights (B, T, T)
        # We use the forward attention for the Association Discrepancy
        attn = _get_attention(self.fwd, x)
        return h, attn


def _get_attention(layer: nn.TransformerEncoderLayer, x: torch.Tensor) -> torch.Tensor:
    """Extract true averaged attention weights from a TransformerEncoderLayer.

    Uses the layer's own MultiheadAttention with need_weights=True so the
    returned matrix reflects the actual learned attention distribution  not a
    scaled-dot-product approximation on normed inputs.

    Note: no torch.no_grad() here — attention must stay in the computation
    graph so that loss_2 (phase 2 maximize AD) can backpropagate through it.
    """
    q = layer.norm1(x) if layer.norm_first else x
    # layer.self_attn is nn.MultiheadAttention(batch_first=True)
    _, attn_weights = layer.self_attn(
        q, q, q,
        need_weights=True,
        average_attn_weights=True,  # average across H heads → (B, T, T)
    )
    return attn_weights  # (B, T, T)


class BiTransformerEncoder(nn.Module):
    def __init__(self, d_model: int = C.D_MODEL, nhead: int = C.NHEAD,
                 n_layers: int = C.N_ENCODER_LAYERS, dropout: float = C.DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([
            _BiTransformerLayer(d_model, nhead, dropout) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attns = []
        for layer in self.layers:
            x, attn = layer(x)
            attns.append(attn)
        # Mean attention across layers → (B, T, T)
        return x, torch.stack(attns, dim=0).mean(0)


# ── BiLSTM ────────────────────────────────────────────────────────────────────

class BiLSTMBridge(nn.Module):
    def __init__(self, input_size: int = C.D_MODEL, hidden_size: int = C.LSTM_HIDDEN,
                 num_layers: int = C.LSTM_LAYERS, dropout: float = C.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(2 * hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, T, 2*hidden)
        return self.proj(out)          # (B, T, d_model)


# ── VAE latent space ──────────────────────────────────────────────────────────

class VAELatent(nn.Module):
    def __init__(self, d_model: int = C.D_MODEL, latent_dim: int = C.LATENT_DIM):
        super().__init__()
        self.mu_head     = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)
        self.decode_proj = nn.Linear(latent_dim, d_model)

    def encode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, d_model)  use mean-pooled representation
        h_pool = h.mean(dim=1)           # (B, d_model)
        return self.mu_head(h_pool), self.logvar_head(h_pool)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def expand(self, z: torch.Tensor, T: int) -> torch.Tensor:
        return self.decode_proj(z).unsqueeze(1).expand(-1, T, -1)


# ── Transformer decoder ───────────────────────────────────────────────────────

class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int = C.D_MODEL, nhead: int = C.NHEAD,
                 n_layers: int = C.N_ENCODER_LAYERS, dropout: float = C.DROPOUT):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.decoder(tgt, memory)


# ── Full BiVAT model ──────────────────────────────────────────────────────────

class BiVAT(nn.Module):
    """
    BiVAT: Bidirectional-Transformer Variational Autoencoder Temporal.

    Inputs:  X   (B, T, d_in)  T = WINDOW_SIZE, d_in = n_indicators
    Outputs: named tuple with
        x_hat     reconstruction  (B, T, d_in)
        mu        latent mean     (B, latent_dim)
        logvar    latent log-var  (B, latent_dim)
        attn      attention map   (B, T, T)  for Association Discrepancy
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = C.D_MODEL,
        nhead: int = C.NHEAD,
        n_layers: int = C.N_ENCODER_LAYERS,
        dropout: float = C.DROPOUT,
        latent_dim: int = C.LATENT_DIM,
        lstm_hidden: int = C.LSTM_HIDDEN,
        lstm_layers: int = C.LSTM_LAYERS,
    ):
        super().__init__()
        d = d_model
        self.input_proj  = nn.Linear(d_in, d)
        self.pe          = SinusoidalPE(d)
        self.bi_encoder  = BiTransformerEncoder(d, nhead, n_layers, dropout)
        self.bi_lstm     = BiLSTMBridge(d, lstm_hidden, lstm_layers, dropout)
        self.vae_latent  = VAELatent(d, latent_dim)
        self.decoder     = TransformerDecoder(d, nhead, n_layers, dropout)
        self.output_proj = nn.Linear(d, d_in)
        self.d_in        = d_in
        self.d_model     = d_model

    def forward(self, x: torch.Tensor) -> dict:
        B, T, _ = x.shape

        # Encoder path
        h = self.pe(self.input_proj(x))           # (B, T, d_model)
        h, attn = self.bi_encoder(h)              # (B, T, d_model), (B, T, T)
        h = self.bi_lstm(h)                       # (B, T, d_model)

        # VAE
        mu, logvar = self.vae_latent.encode(h)    # (B, latent_dim)
        z  = self.vae_latent.reparametrize(mu, logvar)
        z_expanded = self.vae_latent.expand(z, T) # (B, T, d_model)

        # Decoder
        dec_out = self.decoder(z_expanded, h)     # (B, T, d_model)
        x_hat   = self.output_proj(dec_out)       # (B, T, d_in)

        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "attn": attn}

    def forward_encoder_only(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder + BiLSTM → attention map only. Skips VAE and decoder.

        Used in training phase 2 (maximize AD) to avoid the full forward pass
        through VAE and TransformerDecoder, which are not needed for the AD loss.
        """
        h = self.pe(self.input_proj(x))
        _, attn = self.bi_encoder(h)
        return attn  # (B, T, T)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)["x_hat"]
