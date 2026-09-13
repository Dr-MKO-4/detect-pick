"""Constantes partagées entre toutes les couches de l'application."""
import os

BASE_DIR     = os.path.dirname(__file__)
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
XLSX_PATTERN = os.path.join(DATA_DIR, "clean_*_beac_mapping_2SR_*.xlsx")

PAYS = ["cameroun", "congo", "gabon", "guinee_eq", "rca", "tchad"]
PAYS_LABELS: dict[str, str] = {
    "cameroun":  "Cameroun",
    "congo":     "Congo",
    "gabon":     "Gabon",
    "guinee_eq": "Guinée Équatoriale",
    "rca":       "RCA",
    "tchad":     "Tchad",
}
VOLETS = ["Actif", "Passif"]

HOST = os.getenv("BEAC_HOST", "127.0.0.1")
# Source unique du port, en développement (app.py) comme en production
# (main.py) — modifiable sans toucher au code via la variable d'environnement
# BEAC_PORT, pour éviter les collisions avec une autre appli locale.
PORT = int(os.getenv("BEAC_PORT", "8060"))
