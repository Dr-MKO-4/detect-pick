"""
beac_lof/config.py  Constantes et paramètres du pipeline LOF-BEAC.

Toutes les valeurs méthodologiques sont centralisées ici :
modifier ce fichier suffit pour recalibrer le pipeline entier.
"""

from pathlib import Path

# ── Données ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")

PAYS_CEMAC = ["cameroun", "congo", "gabon", "guinee_eq", "rca", "tchad"]
VOLETS = ["Actif", "Passif"]

FILE_MAP = {
    pays: f"clean_{pays}_beac_mapping_2SR"
    for pays in PAYS_CEMAC
}

# ── §2 Prétraitement ─────────────────────────────────────────────────────────

KAPPA: float = 1.4826
"""Facteur de cohérence MAD → σ gaussien (Leys et al. 2013)."""

SHORT_GAP_MAX: int = 3
"""Lacunes ≤ SHORT_GAP_MAX mois consécutifs → interpolation linéaire (MAR)."""

STRUCT_NAN_FRAC: float = 0.50
"""Fraction d'indicateurs manquants dans une période → période structurelle."""

INDICATOR_NAN_MAX: float = 0.20
"""Fraction max imputée en tête de série avant exclusion MNAR."""

LARGE_GAP_MONTHS: int = 6
"""Discontinuité > LARGE_GAP_MONTHS mois en tête → gap structurel détecté."""

# ── §3 STL ───────────────────────────────────────────────────────────────────

STL_PERIOD: int = 12
"""Période saisonnière mensuelle (Cleveland et al. 1990)."""

# ── §4 Réduction de dimensionnalité ──────────────────────────────────────────

PCA_VAR_TARGET: float = 0.90
"""Variance cumulée cible pour le choix de k composantes."""

# ── §5/6 LOF ─────────────────────────────────────────────────────────────────

MINPTS_LB: int = 10
"""Borne inférieure MinPts (Breunig et al. 2000, §6.1)  ≥ 10 pour stabilité."""

MINPTS_UB: int = 20
"""Borne supérieure MinPts  11 valeurs au lieu de 21 → ×2 plus rapide, résultats équivalents."""

# ── §7 Validation ────────────────────────────────────────────────────────────

SYSTEMIC_M_MIN: int = 3
"""Nombre minimal de pays simultanément en anomalie → anomalie systémique CEMAC."""
