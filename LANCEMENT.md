# BEAC Anomaly Detector Guide de lancement

---

## Accès

| Identifiant       | Mot de passe     | Rôle          |
|-------------------|------------------|---------------|
| `analyste.beac`   | `beac2024`       | Analyste      |
| `admin.beac`      | `beacadmin2024`  | Administrateur|

> Pour ajouter un utilisateur ou changer un mot de passe :
> ```python
> import hashlib, sqlite3
> conn = sqlite3.connect("beac.db")
> conn.execute("UPDATE users SET password_hash=? WHERE username=?",
>              (hashlib.sha256(b"nouveaumotdepasse").hexdigest(), "analyste.beac"))
> conn.commit()
> ```

---

## 1. Mise en place (une seule fois)

```powershell
cd m:\Travail\BEAC\detect_pick\
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> Si `Activate.ps1` est bloqué :
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 1b. Pré-entraînement BiVAT (optionnel mais recommandé)

LOF* ne nécessite pas de pré-entraînement (algorithme non-paramétrique).
BiVAT (réseau de neurones) doit être entraîné une fois avant utilisation.

```powershell
# Entraîner toutes les 12 paires (6 pays × 2 volets) — ~20-60 min CPU
python train_all.py

# Entraîner une seule paire
python train_all.py --pays cameroun --volet Actif

# Vérifier que les poids existent
ls models\bivat_*.pt
```

Les poids sont sauvegardés dans `models/bivat_{pays}_{volet}.pt`.
L'application les charge automatiquement sans ré-entraîner.

## 1c. Recherche des meilleurs hyperparamètres (optionnel)

```powershell
# Optimiser LOF* sur un pays
python optimize.py --modele lof --pays cameroun --volet Actif

# Optimiser BiVAT
python optimize.py --modele bivat --pays cameroun --volet Actif
```

Résultats sauvegardés dans `models/hp/`.

---

## 2. Lancer en développement (Dash seul, navigateur)

```powershell
cd m:\Travail\BEAC\detect_pick\
.\.venv\Scripts\Activate.ps1
python app.py
```

- Ouvre l'app dans le **navigateur** à `http://127.0.0.1:8050/`
- Hot-reload Dash actif (`BEAC_DEV=1`)
- **Pas de fenêtre Qt** mode debug Dash uniquement
- Logs : terminal + `logs\beac.log`

```powershell
# Suivre les logs en temps réel
Get-Content -Path "logs\beac.log" -Wait -Tail 50
```

---

## 3. Lancer en production (fenêtre Qt)

```powershell
cd m:\Travail\BEAC\detect_pick\
.\.venv\Scripts\Activate.ps1
python main.py
```

- Ouvre une **fenêtre Windows native** (PyQt6 + QWebEngineView)
- Splash screen BEAC au démarrage
- Serveur Waitress embarqué (stable, pas de debug overlay)
- Double-clic alternatif : `run.bat` à la racine du projet

---

## 4. Construire l'exécutable (.exe)

```powershell
.\.venv\Scripts\Activate.ps1
pyinstaller BEACDetect.spec
```

L'exécutable est généré dans `dist\BEACDetect\BEACDetect.exe`.

> **Important** : copier le dossier `data\` à côté de l'exécutable avant le premier lancement.

---

## 5. Données persistées (BD + figures)

| Emplacement | Contenu |
|---|---|
| `beac.db` | Base SQLite : utilisateurs, sessions, analyses, rapports, préférences, audit |
| `cache\figures\` | Figures Plotly au format JSON rechargées sans relancer le pipeline |
| `logs\beac.log` | Journal applicatif (rotation 5 Mo × 3) |
| `data\` | Fichiers XLSX source (non modifiés par l'application) |
| `models/bivat_*.pt` | Poids BiVAT pré-entraînés par paire (pays × volet) |
| `models/hp/` | Hyperparamètres optimisés (JSON) |

**Préférences utilisateur** : le thème, le pays, le volet et le modèle sélectionnés sont
sauvegardés automatiquement dans `beac.db` (table `user_prefs`) et restaurés à chaque connexion.

**Le comportement est identique en développement et en production.**  
En mode `.exe`, `beac.db` et `cache\` sont créés **à côté du `.exe`**, pas dans le dossier temporaire PyInstaller. Ils survivent aux mises à jour de l'exécutable.

---

## 6. Structure du projet

```
detect_pick/
├── app.py              Point d'entrée Dash (dev navigateur)
├── main.py             Point d'entrée PyQt6 (prod fenêtre native)
├── run.bat             Raccourci double-clic (lance main.py via .venv)
├── config.py           Constantes partagées
├── beac.db             Base de données SQLite (créée au 1er lancement)
├── qt/                 Modules Qt (fenêtre, menubar, splash, aide)
│   ├── window.py       BEACWindow fenêtre principale
│   ├── menubar.py      BEACMenuBar barre de menu draggable
│   ├── splash.py       Splash screen au démarrage
│   └── aide.py         Boîtes de dialogue guide / à propos
├── database/
│   ├── db.py           Connexion SQLite, chemin DB
│   ├── seed.py         Initialisation tables + utilisateurs
│   └── models.py       CRUD (users, sessions, runs, figures, rapports…)
├── pages/              Layouts Dash par page
├── callbacks/          Callbacks Dash (auth, pipeline, navigation…)
├── layout/             Shell, sidebar, icônes SVG
├── beac_lof/           Pipeline LOF*
├── bivat/              Pipeline BiVAT
├── assets/             CSS, logo
│   └── aide/           Fragments HTML du guide (general, lof, bivat…)
├── data/               Fichiers XLSX source
├── cache/figures/      Figures JSON persistées
└── logs/               Journaux applicatifs
```

---

## 7. Dépannage

| Problème | Solution |
|---|---|
| `ModuleNotFoundError` | Vérifier que le venv est activé : `.\.venv\Scripts\Activate.ps1` |
| Page blanche au démarrage | Attendre 3–5 s (serveur Dash démarre en arrière-plan) |
| `beac.db` corrompu | Supprimer `beac.db` la BD est recréée automatiquement (les comptes par défaut sont réinsérés) |
| Figures manquantes (historique) | Vérifier que `cache\figures\` existe et n'a pas été vidé manuellement |
| Mot de passe oublié | Voir section **Accès** ci-dessus pour le réinitialiser via SQLite |
