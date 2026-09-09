# BEAC Detect — Détection d'anomalies BEAC-CEMAC

Application de détection et d'analyse des anomalies dans les données monétaires
de la BEAC (CEMAC). Elle traite les données mensuelles des six pays de la CEMAC
(Cameroun, Congo, Gabon, Guinée équatoriale, RCA, Tchad), séparément pour les
volets **Actif** et **Passif**, et restitue les résultats dans un tableau de
bord interactif (Dash, exécutable en desktop via PyQt6).

L'objectif est d'identifier des observations inhabituelles sans disposer
d'étiquettes d'anomalies déjà validées, tout en donnant à l'analyste des
éléments pour comprendre et vérifier chaque alerte.

## Fonctionnalités

- **Application Dash** utilisable dans un navigateur (`python app.py`) ou en
  application desktop Windows native (PyQt6 + QWebEngineView, `python main.py`
  ou `run.bat`).
- **Authentification** avec rôles analyste/administrateur, stockage SQLite des
  utilisateurs, sessions, analyses, rapports et préférences.
- **Pipeline LOF\*** : chargement XLSX, imputation, standardisation robuste
  médiane/MAD, décomposition STL, RPCA + ACP, score LOF* multi-`MinPts`, seuil
  d'alerte IQR.
- **Pipeline BiVAT** : Transformer bidirectionnel + BiLSTM + autoencodeur
  variationnel temporel sur les résidus LOF, avec prédiction conforme,
  intervalles d'incertitude et explicabilité SHAP.
- **Recherche heuristique d'hyperparamètres** (algorithme génétique maison)
  pour LOF* et BiVAT via `optimize.py`.
- Cache de figures, journalisation, export et préférences utilisateur
  persistées.

## Installation

```powershell
cd detect_pick
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Lancement

```powershell
# Développement (navigateur, hot-reload)
python app.py

# Production (fenêtre desktop native)
python main.py
```

Voir [LANCEMENT.md](LANCEMENT.md) pour le guide complet (entraînement BiVAT,
recherche d'hyperparamètres, packaging `.exe`, dépannage).

## Structure du projet

```
detect_pick/
├── app.py              Point d'entrée Dash (dev navigateur)
├── main.py              Point d'entrée PyQt6 (prod fenêtre native)
├── beac_lof/            Pipeline LOF*
├── bivat/                Pipeline BiVAT
├── optimization/        Recherche d'hyperparamètres (LOF / BiVAT)
├── heuristic_search.py Algorithme génétique générique
├── pages/, callbacks/, layout/, qt/   Interface et logique applicative
├── database/            Connexion SQLite, modèles, seed
├── data/                Fichiers XLSX source par pays
├── models/               Poids BiVAT entraînés par pays/volet
└── Docu/                 Notes méthodologiques (LaTeX)
```

Voir [SPEC.md](SPEC.md) pour la spécification technique détaillée.

## Stack technique

Python 3.10 · Dash / Plotly · PyTorch · PyQt6 + QWebEngineView · SQLite ·
Waitress (serveur de production) · PyInstaller (packaging).

## Documentation

- [LANCEMENT.md](LANCEMENT.md) — installation, lancement, entraînement,
  optimisation, packaging, dépannage.
- [SPEC.md](SPEC.md) — spécification technique.
- [RESUME_PROJET.md](RESUME_PROJET.md) — état d'avancement et priorités.
