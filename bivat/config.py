"""BiVAT hyperparameters  single source of truth (SPEC §4.3)."""

# ── Architecture ──────────────────────────────────────────────────────────────
D_MODEL:          int   = 64
NHEAD:            int   = 4
N_ENCODER_LAYERS: int   = 2
LSTM_HIDDEN:      int   = 64
LSTM_LAYERS:      int   = 2
LATENT_DIM:       int   = 16
DROPOUT:          float = 0.1

# ── Training ──────────────────────────────────────────────────────────────────
WINDOW_SIZE:   int   = 12      # sliding window length (months)
BATCH_SIZE:    int   = 16
EPOCHS:        int   = 200
LR:            float = 1e-3
BETA:          float = 0.5     # KL weight in ELBO
LAMBDA_AD:     float = 0.1     # association discrepancy weight

# ── Train / test split ────────────────────────────────────────────────────────
TRAIN_END:  str = "2019-12-31"   # pre-COVID training window
TEST_START: str = "2020-03-01"

# ── Conformal Prediction ──────────────────────────────────────────────────────
ALPHA_LOW:   float = 0.05   # 95 % coverage
ALPHA_HIGH:  float = 0.10   # 90 % coverage
CAL_SPLIT:   float = 0.20   # fraction for SSBC calibration
MIN_CAL_WINDOWS: int = 20   # nb minimal de fenêtres de calibration exploitables
                            # (sous ce seuil, le calendrier TRAIN_END/TEST_START
                            # ne donne pas assez de points pour une CP fiable —
                            # on replie sur CAL_SPLIT du train, cf. fit_from_lof)

# ── SHAP ──────────────────────────────────────────────────────────────────────
SHAP_N_BG:  int = 50   # background set size for DeepExplainer
SHAP_TOP_N: int = 5    # number of top anomalies to explain

# ── Checkpoint ────────────────────────────────────────────────────────────────
WEIGHTS_PATH: str = "models/bivat_weights.pt"
