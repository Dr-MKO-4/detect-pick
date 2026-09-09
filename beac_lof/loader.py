"""
beac_lof/loader.py  Chargement et parsing des fichiers XLSX (§2.1).

Format source : wide format, indicateurs en lignes, périodes en colonnes (AAAMyy).
Sortie : DataFrame (n_mois × n_indicateurs), index DatetimeIndex, colonnes = IFS Code.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from .config import STRUCT_NAN_FRAC, LARGE_GAP_MONTHS


def _parse_period(col: str) -> Optional[pd.Timestamp]:
    """'2010M1' → Timestamp('2010-01-01'),  autres → None."""
    m = re.match(r"^(\d{4})M(\d{1,2})$", str(col).strip())
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    return None


def _load_raw(path: Path) -> pd.DataFrame:
    """
    Charge un fichier xlsx BEAC et retourne un DataFrame transposé :
    index = DatetimeIndex mensuel, colonnes = codes IFS (uniques, stables).

    Les cellules non numériques (ex. '#VALUE!') sont converties en NaN.
    """
    df_raw = pd.read_excel(path)

    time_cols: dict[str, pd.Timestamp] = {}
    for col in df_raw.columns:
        ts = _parse_period(col)
        if ts is not None:
            time_cols[col] = ts

    if not time_cols:
        raise ValueError(f"Aucune colonne temporelle (format AAAMyy) dans {path.name}")

    data_sub = df_raw[list(time_cols.keys())].copy()
    data_sub.index = df_raw["IFS Code"].values

    # Transposition : périodes (lignes) × indicateurs (colonnes)
    df = data_sub.T.copy()
    df.index = pd.DatetimeIndex([time_cols[c] for c in data_sub.columns])
    df.sort_index(inplace=True)
    df.index.name = "date"

    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _detect_structural_start(df: pd.DataFrame) -> pd.Timestamp:
    """
    Retourne la première date de la fenêtre après troncature structurelle (MNAR tête).

    Règle :
    1. Si une discontinuité > LARGE_GAP_MONTHS mois existe entre les deux premières
       périodes, on retire tout ce qui précède (ex. '2001M12' isolé avant '2010M1').
    2. On avance ensuite jusqu'au premier mois où ≥ 50 % des indicateurs sont renseignés.
    """
    dates = df.index.tolist()

    if len(dates) > 1:
        gap_months = (dates[1] - dates[0]).days / 30.44
        if gap_months > LARGE_GAP_MONTHS:
            df = df.iloc[1:]
            dates = df.index.tolist()

    for dt in dates:
        if df.loc[dt].isna().mean() <= STRUCT_NAN_FRAC:
            return dt

    return dates[0]


def charger_fichier(data_dir: Path, pays: str, volet: str) -> pd.DataFrame:
    """
    Charge, parse et tronque structurellement le fichier (pays, volet).

    Retourne un DataFrame (n_mois × n_indicateurs) à fréquence mensuelle,
    avec les NaN internes préservés (avant imputation).

    Lève FileNotFoundError si le fichier est absent.
    """
    from .config import FILE_MAP
    fname = f"{FILE_MAP[pays.lower()]}_{volet}.xlsx"
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = _load_raw(path)
    start = _detect_structural_start(df)
    df = df.loc[df.index >= start].copy()
    df = df.asfreq("MS")          # fréquence mensuelle (Month Start)
    return df
