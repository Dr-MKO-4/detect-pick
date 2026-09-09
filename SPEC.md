# SPEC  Outil de Détection d'Anomalies BEAC-CEMAC

**Version** : 1.0  2026-08-20  
**Statut** : Spécification technique complète · Phase suivante : Design UI/UX  

---

## 0. Contexte et objectif

L'outil implémente **deux modèles de détection d'anomalies non supervisée** sur les données monétaires BEAC-CEMAC (6 pays, 2 volets Actif/Passif, ~195 observations mensuelles, ~55 indicateurs IFS par fichier) et expose les résultats dans un **dashboard interactif natif desktop** distribué sous forme d'exécutable `.exe` autonome.

### 0.1 Données source

| Paramètre | Valeur |
|-----------|--------|
| Format | XLSX wide format · indicateurs en lignes, périodes en colonnes (`AAAMyy`) |
| Pays | cameroun, congo, gabon, guinee_eq, rca, tchad |
| Volets | Actif, Passif |
| Nom de fichier | `clean_<pays>_beac_mapping_2SR_<volet>.xlsx` |
| Observations | ≈ 195 mois (déc 2001 → mars 2026) |
| Indicateurs | ≈ 55 codes IFS par fichier |
| Unité | Milliards FCFA |
| Ruptures documentées | Choc pétrolier 2014–2016 · Crise camerounaise 2017 · COVID-19 2020–2021 · Post-pandémie 2022–2024 |

Aucune étiquette d'anomalie certifiée → **détection entièrement non supervisée**.

---

## 1. Stack technique

| Couche | Choix | Justification |
|--------|-------|---------------|
| Fenêtre native | **PyQt6** | Taskbar OS, menus natifs, DPI-awareness, packagable PyInstaller |
| Dashboard web | **Dash (Plotly)** servi sur `localhost:8050` | Figures interactives, dropdowns, animations |
| Intégration | **QWebEngineView** (inclus PyQt6) | Embed Dash dans la fenêtre Qt sans browser externe |
| ML / Algèbre | **PyTorch 2.x** | GPU/CPU unifié · déjà utilisé dans `beac_lof/` · requis pour BiVAT |
| Décomposition STL | **statsmodels** | STL Python standard · `stl.STL(robust=True, period=12)` |
| SHAP | **shap** (`DeepExplainer`) | Auditabilité réglementaire COBAC |
| Packaging | **PyInstaller** | `.exe` autonome · hook PyQt6 + QWebEngine inclus |

### 1.1 Optimisation matérielle automatique

Détectée au démarrage et appliquée à toutes les opérations PyTorch :

```python
import os, torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    # torch.compile() activé pour les modèles BiVAT
elif torch.backends.mps.is_available():       # Apple Silicon
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
    torch.set_num_threads(os.cpu_count())
```

AMP (`torch.cuda.amp.autocast`) activé uniquement sur CUDA.

### 1.2 Mode développement (sans PyQt6)

PyQt6 n'est nécessaire **qu'en phase de packaging**. Pour le développement quotidien, Dash tourne seul dans le navigateur :

```bash
# Lance uniquement le serveur Dash  ouvre http://127.0.0.1:8050
python app.py
```

```python
# app.py  switch dev/prod
import os
DEV = os.getenv("BEAC_DEV", "1") == "1"

if __name__ == "__main__":
    app.run(
        debug=DEV,          # hot-reload + DevTools si DEV=1
        host="127.0.0.1",
        port=8050,
    )
```

| Variable | Valeur | Effet |
|----------|--------|-------|
| `BEAC_DEV=1` (défaut) | dev | Hot-reload · erreurs inline · pas de PyQt6 requis |
| `BEAC_DEV=0` | prod | Pas de hot-reload · lancé depuis `main.py` dans QWebEngineView |

Workflow développement type :
1. `python app.py` → navigateur Chrome/Firefox sur `localhost:8050`
2. Éditer `app.py`, `beac_lof/graphiques.py`, `bivat/graphiques.py` → rechargement automatique
3. PyQt6 uniquement testé en fin de sprint avant packaging `.exe`

---

## 2. Structure cible du projet (après nettoyage)

```
detect_pick/
├── SPEC.md                        ← ce fichier
├── main.py                        ← point d'entrée PyQt6 + Dash (prod)
├── app.py                         ← layout Dash + callbacks (dev + prod)
│
├── asset/
│   ├── logo_beac.jfif             ← logo BEAC (EXISTANT)
│   └── icon.ico                   ← icône .exe (à générer depuis logo)
│
├── beac_lof/                      ← LOF pipeline complet (EXISTANT, validé)
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── preprocessing.py
│   ├── decomposition.py
│   ├── reduction.py
│   ├── detection.py
│   ├── pipeline.py
│   └── graphiques.py              ← Figs 1–14, A, B (21 figures LOF)
│
├── bivat/                         ← BiVAT pipeline (À CRÉER)
│   ├── __init__.py
│   ├── config.py
│   ├── architecture.py
│   ├── training.py
│   ├── scoring.py
│   ├── explainability.py
│   ├── pipeline.py
│   └── graphiques.py              ← Figs E, D, E bis, F, G (5 figures BiVAT)
│
├── utils/
│   ├── hardware.py                ← détection DEVICE au démarrage
│   └── export.py                  ← export PNG / CSV / Excel
│
├── Design application Windows/    ← maquettes (EXISTANT, référence)
├── Docu/                          ← documents méthodologiques LaTeX (EXISTANT)
└── data/                          ← fichiers XLSX source (non versionné)
```

---

## 3. Modèle 1  LOF (état : COMPLET, validé)

### 3.1 Pipeline en 6 étapes (`beac_lof/`)

Chaque fichier `(pays, volet)` est traité **indépendamment** (pas de pooling CEMAC).

| Étape | Module | Implémentation |
|-------|--------|----------------|
| 1 · Chargement | `loader.py` · `charger_fichier()` | Parse `AAAMyy`, transpose, détecte début structurel |
| 2 · Imputation | `preprocessing.py` · `imputer()` | Interpolation linéaire (MAR ≤ 3 mois) → MNAR exclusion (> 20% tête) → MICE PyTorch (OLS `torch.linalg.lstsq`, 10 iter) |
| 3 · Normalisation | `preprocessing.py` · `mad_standardize()` | Médiane/MAD · κ = 1.4826 · `torch.nanmedian` · fallback std si MAD = 0 |
| 4 · STL | `decomposition.py` · `stl_decompose()` | `statsmodels.STL(robust=True, period=12)` · résidu R seul retenu |
| 5 · RPCA + ACP | `reduction.py` | RPCA-IALM (Candès 2011) λ = 1/√max(m,n) · ACP SVD → k compos. pour 90% variance |
| 6 · LOF* + τ | `detection.py` | LOF*(p) = max{LOF_k(p) \| k ∈ [10,30]} · τ = Q3 + 1.5×IQR |

**Implémentation entièrement PyTorch**  aucune boucle numpy sur les observations.

### 3.2 Inventaire complet des figures  21 figures (`graphique.tex`)

Le document `graphique.tex` définit **21 figures** réparties sur 4 phases du workflow.  
Statut code : ✅ implémenté · ⚠️ implémenté avec écart · ❌ à créer

#### Phase 1  Diagnostic des données (commun LOF + BiVAT)

| Fig | Titre | Module cible | Statut | Priorité |
|-----|-------|-------------|--------|----------|
| 1 | Carte des données manquantes | `beac_lof/graphiques.py` | ✅ | Essentiel |
| 2 | Heatmap corrélations résidus STL | `beac_lof/graphiques.py` | ✅ | Essentiel |
| 3 | Décomposition RobustSTL 4 panneaux | `beac_lof/graphiques.py` | ✅ | Essentiel |
| A | Variance expliquée RPCA sur fenêtres glissantes de 36 mois | `beac_lof/graphiques.py` | ❌ | Important |

**Fig. A**  Courbe de variance cumulée des 5 premières composantes RPCA calculée sur des fenêtres glissantes de 36 mois (pas = 1 mois). Annotations des ruptures documentées (choc pétrolier 2015, COVID 2020). Ligne horizontale à 90%. Détecte les changements de structure de covariance dans le temps (non-stationnarité).

#### Phase 2  Calibration et validation interne

| Fig | Titre | Module cible | Statut | Priorité |
|-----|-------|-------------|--------|----------|
| 4 | Scree plot RPCA | `beac_lof/graphiques.py` | ✅ | Essentiel |
| 5 | Biplot ACP (composantes 1 et 2) | `beac_lof/graphiques.py` | ✅ | Complémentaire |
| E | Intervalles confiance BiVAT  fenêtre calibration | `bivat/graphiques.py` | ❌ | Essentiel |

**Fig. E (calibration)**  Score BiVAT + enveloppes CP à 90% et 95% sur la fenêtre d'entraînement (déc 2001 – déc 2019). Quantile empirique $\hat{q}_{1-\alpha}$. Bande grisée sur les périodes à CP pondérée.

#### Phase 3  Production et interprétation des résultats

| Fig | Titre | Module cible | Statut | Priorité |
|-----|-------|-------------|--------|----------|
| 6 | Projection UMAP colorée par LOF* | `beac_lof/graphiques.py` | ✅ | Complémentaire |
| 7 | Profils LOF(MinPts)  top-20 anomalies | `beac_lof/graphiques.py` | ✅ | Essentiel |
| 8 | Distribution empirique LOF* + seuils τ | `beac_lof/graphiques.py` | ✅ | Essentiel |
| 9 | Boxplot LOF* par année | `beac_lof/graphiques.py` | ✅ | Important |
| 10 | Série temporelle LOF*(t) | `beac_lof/graphiques.py` | ✅ | Essentiel |
| 11 | Superposition LOF* + résidu STL + série brute | `beac_lof/graphiques.py` | ✅ | Essentiel |
| 12 | **Bar chart horizontal** indicateurs contributeurs | `beac_lof/graphiques.py` | ⚠️ | Essentiel |
| 13 | Heatmap anomalies (indicateurs × mois anomaux) | `beac_lof/graphiques.py` | ✅ | Essentiel |
| 14 | Carte de chaleur inter-pays (animée) | `beac_lof/graphiques.py` | ✅ | Essentiel |
| B | Concordance Actif–Passif par pays | `beac_lof/graphiques.py` | ❌ | Essentiel |

> **⚠️ Fig. 12** : le code actuel (`fig12_radar_chart`) produit un **radar chart**. Le document spécifie un **bar chart horizontal** trié par contribution absolue décroissante, 15 indicateurs max, barres colorées (rouge = contribution positive à l'anomalie, bleu = négative). À corriger.

**Fig. B**  Pour chaque pays : 2 panneaux empilés (LOF* Actif / LOF* Passif) sur le même axe temporel, même seuil τ. Zones rouges = anomalie concordante (les deux volets > τ) → événement économique réel. Zones oranges = anomalie discordante (un seul volet > τ) → vérification comptable requise. 6 graphiques (un par pays).

#### Phase 4  Comparaison inter-modèles et décision finale

| Fig | Titre | Module cible | Statut | Priorité |
|-----|-------|-------------|--------|----------|
| D | Comparaison LOF vs BiVAT dans le temps | `bivat/graphiques.py` | ❌ | Essentiel |
| E bis | Score BiVAT + intervalles CP  fenêtre test | `bivat/graphiques.py` | ❌ | Essentiel |
| F | Attribution SHAP top-5 anomalies BiVAT | `bivat/graphiques.py` | ❌ | Essentiel |
| G | Carte de confusion LOF/BiVAT par pays et type | `bivat/graphiques.py` | ❌ | Important |

**Fig. D**  Deux courbes normalisées [0,1] sur la fenêtre de test (mars 2020 – mars 2026) : LOF*(t) en bleu, score BiVAT en rouge. 4 zones colorées : vert pâle (accord normalité), rouge pâle (accord anomalie = alertes niveau 1), orange gauche (LOF seul = anomalie ponctuelle), orange droit (BiVAT seul = anomalie contextuelle/collective non vue par LOF).

**Fig. E bis**  Score BiVAT $s^{\mathrm{discrim}}_t$ sur la fenêtre de test avec enveloppes CP à 90% et 95% (rolling-origin). Bande grisée COVID-19 (mars 2020 – déc 2021) où la CP pondérée est active. Classification : anomalie certaine (score + borne inférieure > seuil) vs anomalie ambiguë.

**Fig. F**  5 bar charts horizontaux (un par anomalie BiVAT majeure). 15 indicateurs triés par $|\phi_d|$ décroissant. Barres rouges ($\phi_d > 0$) et bleues ($\phi_d < 0$). Valeurs en % de contribution totale. Sous-titre : mois, score BiVAT, score LOF correspondant. Produit par DeepSHAP (`bivat/explainability.py`).

**Fig. G**  Heatmap 6×4 : 6 pays CEMAC × 4 types d'anomalie (ponctuelle, contextuelle, collective, systémique). Couleur = taux de concordance LOF–BiVAT. Colormap blanc (0%) → vert foncé (100%). Encadré si concordance > 70%. Valeurs numériques par cellule (% + nombre d'alertes).

---

## 4. Modèle 2  BiVAT (état : À CRÉER)

BiVAT = **Bi**directional-Transformer **V**ariational **A**uto-encoder + **T**emporal.  
Répond aux lacunes L1–L5 du LOF (dépendances temporelles, dégénérescence haute dimension, incertitude quantifiée, explicabilité par indicateur, instabilité MinPts).

### 4.1 Choix architecturaux (D1–D4)

#### D1  Encodeur hybride Bi-Transformer + BiLSTM

Justification : le Transformer pur est faible sur N < 200 (manque de données pour apprendre l'attention) ; BiLSTM compense en capturant les dynamiques locales.

```
Entrée X (T×d) → Bi-Transformer (L=2 couches, H=4 têtes, d_model=64)
                → BiLSTM (hidden=64, 2 couches)
                → μ_z, log σ²_z  (espace latent z, dim=16)
                → Reparameterization trick → z
                → Décodeur Transformer → X̂ (T×d)
```

- `d_model = 64`, `nhead = 4`, `num_encoder_layers = 2`
- `BiLSTM hidden = 64`, `num_layers = 2`, `dropout = 0.1`
- `latent_dim = 16`
- Positional encoding sinusoïdal (Vaswani 2017)

#### D2  Conformal Prediction (couverture formelle)

Garantie : Pr(ŷ_t ∈ C_α(x_t)) ≥ 1 − α pour tout t.

- **SSBC** (Split Conformal + Bootstrapped Calibration) pour N < 200 : split 80/20, calibration sur 20%, intervalles de prédiction à α = 0.05 et 0.10
- **EnbPI** (Ensemble Batch Prediction Intervals) pour non-stationnarité : adaptation en ligne des intervalles via résidus récents

Implémentation dans `scoring.py` :
```python
def conformal_intervals(model, X_cal, y_cal, alpha=0.05) -> tuple[np.ndarray, np.ndarray]:
    scores = np.abs(y_cal - model.predict(X_cal))
    q_hat = np.quantile(scores, (1 - alpha) * (1 + 1/len(scores)))
    return y_pred - q_hat, y_pred + q_hat
```

#### D3  SHAP DeepExplainer (auditabilité COBAC)

Complexité O(d·T) = O(55 × 195) = O(10 725)  tractable.

- `shap.DeepExplainer(model, X_background)` sur le sous-ensemble des mois anomaux
- Output : matrice d'attributions (T_anomalies × d) → classement des indicateurs contributeurs
- Exposé dans la Fig G (radar SHAP)

#### D4  Score hybride Reconstruction × Association Discrepancy

Xu et al. 2022 (Anomaly Transformer) : +18.82 pts F1 vs reconstruction seule (Table 3 ablation).

```
Score(t) = Reconstruction_error(t) × Association_Discrepancy(t)
```

- `Reconstruction_error(t) = ||X_t - X̂_t||²`
- `Association_Discrepancy(t) = KL_symétrique(P_t || Q_t)` où P = distribution d'attention apprise, Q = prior gaussien de l'attention

Entraînement en 2 phases minimax :
- Phase 1 (minimize) : minimise ELBO + Assoc. Disc. → anomalies "repoussées"
- Phase 2 (maximize) : maximise Assoc. Disc. → contraste amplifié

### 4.2 Fonction de perte ELBO complète

```
L = E[||X - X̂||²]                         ← reconstruction MSE
  - β · KL(q(z|X) || p(z))                  ← régularisation VAE (β = 0.5)
  + λ_ad · AssocDiscrepancy(Attn, Prior)     ← association discrepancy (λ_ad = 0.1)
```

### 4.3 Hyperparamètres `bivat/config.py`

```python
# Architecture
D_MODEL:       int   = 64
NHEAD:         int   = 4
N_ENCODER_LAYERS: int = 2
LSTM_HIDDEN:   int   = 64
LSTM_LAYERS:   int   = 2
LATENT_DIM:    int   = 16
DROPOUT:       float = 0.1

# Entraînement
WINDOW_SIZE:   int   = 12      # fenêtre glissante T=12 mois
BATCH_SIZE:    int   = 16
EPOCHS:        int   = 200
LR:            float = 1e-3
BETA:          float = 0.5     # poids KL divergence
LAMBDA_AD:     float = 0.1     # poids association discrepancy

# Conformal
ALPHA_IQR:     float = 0.05
ALPHA_P95:     float = 0.10
CAL_SPLIT:     float = 0.20    # fraction calibration SSBC

# SHAP
SHAP_N_BG:     int   = 50      # taille background set
```

### 4.4 Figures BiVAT

Les figures BiVAT sont détaillées dans l'inventaire complet (section 3.2) avec leur positionnement exact dans le workflow. Résumé :

| Fig | Phase | Module | Description |
|-----|-------|--------|-------------|
| E (calibration) | Phase 2 | `bivat/graphiques.py` | Score BiVAT + intervalles CP fenêtre calibration |
| E bis | Phase 4 | `bivat/graphiques.py` | Score BiVAT + intervalles CP fenêtre test |
| D | Phase 4 | `bivat/graphiques.py` | Comparaison LOF vs BiVAT normalisés [0,1] · 4 zones |
| F | Phase 4 | `bivat/graphiques.py` | Bar chart SHAP horizontal · top-5 anomalies BiVAT |
| G | Phase 4 | `bivat/graphiques.py` | Heatmap confusion LOF/BiVAT · 6 pays × 4 types |

---

## 5. Application desktop  Architecture

### 5.1 Point d'entrée `main.py`

```python
import sys, threading
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
import app as dash_app

def start_dash():
    dash_app.server.run(host="127.0.0.1", port=8050, debug=False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BEAC Anomaly Detector")
        self.setMinimumSize(1400, 900)
        view = QWebEngineView()
        view.setUrl(QUrl("http://127.0.0.1:8050"))
        self.setCentralWidget(view)

if __name__ == "__main__":
    t = threading.Thread(target=start_dash, daemon=True)
    t.start()
    qt_app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(qt_app.exec())
```

### 5.2 Layout Dash `app.py`

4 phases (onglets / sidebar) conformes à `Docu/graphique.tex` :

| Phase | Figures | Modèle | Priorité |
|-------|---------|--------|----------|
| **Phase 1** · Diagnostic des données | Fig. 1, 2, 3, A | Commun | Essentiel / Important |
| **Phase 2** · Calibration interne | Fig. 4, 5 (LOF) · Fig. E calibration (BiVAT) | LOF + BiVAT | Essentiel |
| **Phase 3** · Résultats et interprétation | Fig. 6, 7, 8, 9, 10, 11, 12, 13, 14 (LOF) · Fig. B (LOF) | LOF | Essentiel |
| **Phase 4** · Comparaison et décision | Fig. D, E bis, F, G | Comparaison | Essentiel / Important |

> Les figures de Phase 2 ne sont affichées que si le modèle correspondant a été entraîné/calibré.  
> Fig. B est produite après calcul des scores LOF* sur les 12 fichiers (6 pays × 2 volets).  
> Fig. D, E bis, F, G nécessitent que les deux modèles aient été exécutés sur la fenêtre de test.

**Contrôles globaux** (sidebar gauche) :
- Sélecteur pays (multi-select + "Tous")
- Sélecteur volet (Actif / Passif / Les deux)
- Sélecteur modèle (LOF / BiVAT / Comparaison)
- Bouton "Lancer l'analyse" → déclenche le pipeline sélectionné
- Barre de progression (callback Dash interval)
- Résumé : n anomalies détectées / seuil τ

### 5.3 Callbacks Dash principaux

```python
@app.callback(
    [Output("store-lof-results", "data"),
     Output("progress-bar", "value")],
    Input("btn-run", "n_clicks"),
    [State("dropdown-pays", "value"),
     State("dropdown-volet", "value"),
     State("dropdown-modele", "value")]
)
def run_pipeline(n_clicks, pays_list, volet, modele):
    # Lance PipelineLOF ou PipelineBiVAT selon sélection
    ...
```

---

## 6. Design UI/UX

Référence : `Design application Windows/BEAC Detector.dc.html`  
Logo : `asset/logo_beac.jfif`

### 6.1 Tokens CSS  deux thèmes dans `assets/style.css`

Le bouton `[☀]` dans la titlebar bascule entre thème sombre et clair via `data-theme` sur `<html>`.

#### Thème sombre (défaut)  `BEAC Detector.dc.html`

```css
:root {
  /* Fonds */
  --bg:      #050505;
  --surf:    #0f0f0f;
  --surf2:   #181818;
  --surf3:   #222222;

  /* Bordures */
  --border:  rgba(255,255,255,.08);
  --border2: rgba(255,255,255,.14);

  /* Accent institutionnel BEAC */
  --gold:    #D4A020;
  --gold-lt: rgba(212,160,32,.18);
  --gold-md: rgba(212,160,32,.35);

  /* Texte */
  --text:   #f0f0f0;
  --muted:  rgba(240,240,240,.45);
  --muted2: rgba(240,240,240,.22);

  /* Statuts */
  --red:      #E05252;
  --red-lt:   rgba(224,82,82,.15);
  --green:    #52C97E;
  --green-lt: rgba(82,201,126,.12);
  --blue:     #5BA8E5;

  /* Typographie */
  --font-body:    'Roboto', sans-serif;
  --font-mono:    'Roboto Mono', monospace;
}
```

#### Thème clair  `styles.css` (classical)

```css
[data-theme="light"] {
  /* Fonds */
  --bg:      #f3f2f2;
  --surf:    #eae9e9;
  --surf2:   #e0dfdf;
  --surf3:   #d7d3d3;

  /* Bordures */
  --border:  rgba(32,31,29,.12);
  --border2: rgba(32,31,29,.20);

  /* Accent BEAC (or ambré, palette claire) */
  --gold:    #b68235;
  --gold-lt: rgba(182,130,53,.12);
  --gold-md: rgba(182,130,53,.28);

  /* Texte */
  --text:   #201f1d;
  --muted:  rgba(32,31,29,.55);
  --muted2: rgba(32,31,29,.30);

  /* Statuts */
  --red:      #c0392b;
  --red-lt:   rgba(192,57,43,.10);
  --green:    #27ae60;
  --green-lt: rgba(39,174,96,.10);
  --blue:     #2980b9;

  /* Typographie */
  --font-body:    'Lora', serif;
  --font-mono:    'Roboto Mono', monospace;
  --font-heading: 'Cormorant Garamond', serif;
}
```

**Bascule JS dans Dash** :
```python
app.clientside_callback(
    """function(n) {
        const html = document.documentElement;
        const next = html.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
        html.setAttribute('data-theme', next);
        return next === 'light' ? '🌙' : '☀';
    }""",
    Output("btn-theme", "children"),
    Input("btn-theme", "n_clicks"),
    prevent_initial_call=True,
)

### 6.2 Écrans définis dans la maquette (5 écrans)

| Écran | Titre | Description |
|-------|-------|-------------|
| 01 | Connexion | Formulaire login · logo BEAC · fond grille subtile · bande gold · accès réservé agents BEAC |
| 02 | Données · chargement & prévisualisation | Drag & drop XLSX · liste 12 fichiers avec statut ✓/… · 4 stat-cards (195 obs, 55 ind, % manquants, début) · table prévisualisation sticky |
| 03 | Analyse en cours | Sidebar + topbar phases 1-2-3-4 · Fig. 10 pleine largeur + Fig. 8 + Fig. 13 · bouton "Arrêter" rouge · barre de progression 64% · badge "● Live" animé |
| 04 | Rapport | Config rapport (type, pays, période, modèles, format PDF/Excel/HTML) · historique des rapports · aperçu PDF embarqué |
| 05 | Modèles · réentraînement | Sidebar collapsée (48px icônes seules) · carte LOF* (calibré ✓) · carte BiVAT (⚠ non entraîné) · journal d'entraînement terminal-style Roboto Mono |

### 6.3 Shell de l'application

```
┌─ Titlebar 32px ─────────────────────────────────────────────────────┐
│ ● ● ●   BEAC Anomaly Detector  v1.0                          [☀]  │
├─ Menubar 26px ──────────────────────────────────────────────────────┤
│ Fichier  Édition  Sélection  Affichage  Aide                        │
├─ Sidebar 220px ──┬─ Main ──────────────────────────────────────────┤
│ [logo BEAC]      │ Topbar 44px : Section › Page              ctx   │
│                  ├────────────────────────────────────────────────  │
│ PRINCIPAL        │                                                   │
│ > Données        │   Zone de contenu (scrollable)                   │
│   Analyse        │   Grille de panels + figures Plotly              │
│   Rapport        │                                                   │
│                  │                                                   │
│ AVANCÉ           │                                                   │
│   Modèles        │                                                   │
│   Paramètres     │                                                   │
│   Mode d'emploi  │                                                   │
│                  │                                                   │
│ ─────────────── │                                                   │
│ analyste.beac    │                                                   │
└──────────────────┴────────────────────────────────────────────────-─┘
```

### 6.4 Composants Dash/CSS à implémenter

| Composant | CSS class | Description |
|-----------|-----------|-------------|
| Panel | `.panel` / `.panel-header` / `.panel-body` | Conteneur graphique · bordure `var(--border)` · fond `var(--surf)` |
| Bouton primaire | `.btn-gold` | Bordure + texte `var(--gold)` · hover fond `var(--gold-lt)` |
| Bouton action | `.btn-solid-gold` | Fond `var(--gold)` · texte noir · action principale |
| Bouton discret | `.btn-ghost` | Texte `var(--muted)` · hover border blanc |
| Select | `.select` | Fond `var(--surf2)` · focus border `var(--gold)` |
| Tag anomalie | `.tag-red` | Fond `var(--red-lt)` · texte `var(--red)` |
| Tag normal | `.tag-green` | Fond `var(--green-lt)` · texte `var(--green)` |
| Tag info | `.tag-gold` | Fond `var(--gold-lt)` · texte `var(--gold)` |
| Stat card | `.stat-card` | Chiffre clé en `var(--gold)` 26px bold · label muted 10px |
| Progress bar | `.progress-track` / `.progress-fill` | Hauteur 3px · fond `var(--surf3)` |
| Table données | `.data-table` | Entêtes uppercase 9px · valeurs Roboto Mono 10.5px |
| Nav item actif | `.nav-item.active` | Fond `var(--gold-lt)` · bordure `var(--gold-md)` · texte `var(--gold)` |

### 6.5 Plotly  thème cohérent avec le design

```python
PLOTLY_THEME = {
    "paper_bgcolor": "#0f0f0f",   # var(--surf)
    "plot_bgcolor":  "#0f0f0f",
    "font":          {"color": "#f0f0f0", "family": "Roboto, sans-serif", "size": 11},
    "gridcolor":     "rgba(255,255,255,.06)",
    "colorway":      ["#D4A020", "#5BA8E5", "#52C97E", "#E05252", "#9B59B6"],
}

# Appliqué à chaque figure via :
fig.update_layout(**PLOTLY_THEME)
fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", showline=False)
fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", showline=False)
```

---

## 7. Packaging PyInstaller

```bash
pyinstaller main.py \
    --name BEACDetect \
    --onefile \
    --windowed \
    --add-data "data;data" \
    --hidden-import PyQt6.QtWebEngineWidgets \
    --hidden-import dash \
    --hidden-import plotly \
    --hidden-import torch \
    --collect-all statsmodels \
    --collect-all shap \
    --icon assets/icon.ico
```

Points d'attention PyInstaller :
- Qt WebEngine nécessite `--collect-all PyQt6` ou hook manuel pour les `.so`/`.dll`
- PyTorch : exclure CUDA si CPU-only build (`--exclude-module torch.cuda`)
- Données : `data/` doit être embarqué ou chargé depuis chemin relatif via `sys._MEIPASS`

---

## 7. État d'avancement

| Composant | État | Module |
|-----------|------|--------|
| Loader XLSX | ✅ Complet | `beac_lof/loader.py` |
| Prétraitement (MICE, MAD) | ✅ Complet | `beac_lof/preprocessing.py` |
| STL | ✅ Complet | `beac_lof/decomposition.py` |
| RPCA-IALM + ACP | ✅ Complet | `beac_lof/reduction.py` |
| LOF* + seuil τ | ✅ Complet | `beac_lof/detection.py` |
| Pipeline LOF orchestrateur | ✅ Complet | `beac_lof/pipeline.py` |
| Figs 1–3, 4–14 LOF (13 correctes) | ✅ Complet | `beac_lof/graphiques.py` |
| Fig. 12 (bar chart → actuellement radar) | ⚠️ À corriger | `beac_lof/graphiques.py` |
| Fig. A (variance RPCA glissante) | ❌ À créer | `beac_lof/graphiques.py` |
| Fig. B (concordance Actif–Passif) | ❌ À créer | `beac_lof/graphiques.py` |
| Architecture BiVAT | ❌ À créer | `bivat/architecture.py` |
| Entraînement BiVAT | ❌ À créer | `bivat/training.py` |
| Score hybride + Conformal | ❌ À créer | `bivat/scoring.py` |
| SHAP | ❌ À créer | `bivat/explainability.py` |
| Pipeline BiVAT | ❌ À créer | `bivat/pipeline.py` |
| Fig. E calibration (BiVAT) | ❌ À créer | `bivat/graphiques.py` |
| Fig. D (comparaison LOF vs BiVAT) | ❌ À créer | `bivat/graphiques.py` |
| Fig. E bis (score BiVAT + CP test) | ❌ À créer | `bivat/graphiques.py` |
| Fig. F (SHAP bar chart) | ❌ À créer | `bivat/graphiques.py` |
| Fig. G (confusion LOF/BiVAT) | ❌ À créer | `bivat/graphiques.py` |
| App Dash layout | ❌ À créer | `app.py` |
| Fenêtre PyQt6 | ❌ À créer | `main.py` |
| Utilitaires hardware/export | ❌ À créer | `utils/` |
| Packaging `.exe` | ❌ À créer |  |

**Avancement estimé : 40%**

---

## 8. Références méthodologiques

- **Breunig et al. 2000**  LOF: Identifying Density-Based Local Outliers
- **Candès et al. 2011**  Robust PCA via Convex Optimization
- **Cleveland et al. 1990**  STL: A Seasonal-Trend Decomposition
- **Leys et al. 2013**  Detecting outliers: Do not use standard deviation, use MAD
- **Xu et al. 2022**  Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy
- **Kingma & Welling 2013**  Auto-Encoding Variational Bayes
- **Angelos et al. 2023**  Conformal Prediction for Time Series (SSBC, EnbPI)
- **Lundberg & Lee 2017**  SHAP Unified Approach to Explaining Model Outputs
- **Zimek et al. 2012**  A survey on unsupervised outlier detection in high-dimensional numerical data

---

## 9. Phase suivante : Design UI/UX

Décisions à prendre pour la phase design :
1. **Palette de couleurs** : thème BEAC institutionnel (vert/blanc ?) vs. sombre analytique
2. **Layout sidebar** : fixe à gauche (largeur 280px) ou collapsible
3. **Navigation** : onglets horizontaux (phases 1–4) ou accordéon vertical
4. **Figures** : taille minimale / comportement responsive dans QWebEngineView
5. **Chargement** : skeleton loading pendant calcul pipeline (peut durer 30–90s)
6. **Export** : bouton par figure (PNG/HTML) + export global CSV/Excel anomalies

> **Note** : Dash est rendu dans un navigateur embarqué Chromium (QWebEngineView)  CSS standard et flexbox s'appliquent normalement. Éviter les dépendances JS externes non packagées.
