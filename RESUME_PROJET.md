# Projet detect_pick — Détection d'anomalies BEAC-CEMAC

## Objectif

`detect_pick` est une application de détection et d'analyse des anomalies dans les données monétaires BEAC-CEMAC. Elle traite les données mensuelles des six pays de la CEMAC, séparément pour les volets Actif et Passif, puis restitue les résultats dans un tableau de bord interactif.

L'objectif est d'identifier des observations inhabituelles sans disposer d'étiquettes d'anomalies déjà validées, tout en donnant à l'analyste des éléments pour comprendre et vérifier les alertes.

## Ce qui est réalisé

### Application et interface

- Application Dash utilisable dans un navigateur en développement avec `python app.py`.
- Application desktop Windows prévue avec PyQt6, QWebEngineView et Dash, lancée avec `python main.py` ou `run.bat`.
- Authentification, rôles analyste/administrateur et stockage SQLite des utilisateurs, sessions, analyses, rapports et préférences.
- Pages dédiées aux données, à l'analyse, aux rapports, aux modèles, à l'optimisation et aux paramètres.
- Cache des figures, journalisation et sauvegarde des préférences utilisateur.n

### Pipeline LOF*

Le pipeline LOF est la partie la plus avancée et sert de référence principale. Pour chaque couple pays/volet, il enchaîne :

1. chargement et remise en forme des fichiers XLSX ;
2. traitement des valeurs manquantes et exclusion des indicateurs trop incomplets ;
3. standardisation robuste par médiane/MAD ;
4. décomposition STL avec saisonnalité mensuelle ;
5. décomposition RPCA puis réduction ACP ;
6. calcul du score LOF* sur plusieurs valeurs de `MinPts` ;
7. calcul d'un seuil d'alerte basé sur l'IQR.

Les résultats intermédiaires, les scores, les seuils, les indicateurs contributeurs et les graphiques sont conservés pour l'interprétation et l'export.

### Pipeline BiVAT

Un second modèle est intégré pour compléter LOF* : BiVAT combine un Transformer bidirectionnel, un BiLSTM et un autoencodeur variationnel temporel. Il exploite les résidus normalisés produits par LOF, entraîne ou recharge un modèle par pays/volet, calcule un score discriminant et ajoute :

- une séparation entraînement/calibration/test ;
- la prédiction conforme et des intervalles d'incertitude ;
- l'explicabilité des alertes par SHAP ;
- la comparaison future des alertes LOF* et BiVAT.

Les fichiers de poids sont enregistrés dans `models/` et peuvent être générés avec `train_all.py`.

## Livrables présents

- `app.py` : entrée Dash pour le développement ;
- `main.py` : entrée de l'application desktop ;
- `beac_lof/` : pipeline LOF complet ;
- `bivat/` : architecture, entraînement, scoring et explicabilité BiVAT ;
- `callbacks/`, `pages/`, `layout/` et `qt/` : interface et logique applicative ;
- `data/` : fichiers sources XLSX ;
- `models/` : poids entraînés et hyperparamètres ;
- `cache/figures/`, `logs/` et `beac.db` : données persistées par l'application ;
- `SPEC.md` : spécification technique détaillée ;
- `LANCEMENT.md` : installation, lancement, entraînement et packaging.

## Étapes suivantes

### Priorité 1 — Stabiliser et valider

- Tester LOF* sur les 12 couples pays/volet et conserver un jeu de résultats de référence.
- Vérifier les données source, les périodes couvertes et les indicateurs exclus pour chaque fichier.
- Ajouter des tests reproductibles sur l'imputation, la STL, la RPCA, l'ACP, le score LOF* et le seuil `tau`.
- Tester les cas d'erreur de l'interface : fichier absent, données insuffisantes, poids BiVAT manquants et pipeline interrompu.

### Priorité 2 — Finaliser BiVAT

- Entraîner et archiver les 12 modèles BiVAT avec une configuration documentée.
- Contrôler la stabilité des scores selon la graine, la fenêtre temporelle et les hyperparamètres.
- Vérifier les intervalles de prédiction conforme et les résultats SHAP sur des anomalies connues ou documentées.
- Comparer les alertes LOF* et BiVAT pour distinguer les anomalies ponctuelles, contextuelles, collectives et systémiques.

### Priorité 3 — Compléter les analyses et les graphiques

- Créer les figures encore manquantes : variance RPCA glissante, concordance Actif–Passif et comparaison LOF/BiVAT.
- Remplacer le radar de la figure 12 par le bar chart horizontal spécifié.
- Ajouter les exports CSV/Excel/PDF nécessaires à l'exploitation par les analystes.
- Documenter une procédure de revue humaine des alertes et de validation économique/comptable.

### Priorité 4 — Préparer la diffusion

- Tester l'application en mode production avec PyQt6 et Waitress.
- Construire l'exécutable avec PyInstaller et vérifier les chemins de `data/`, `models/`, `cache/` et `beac.db`.
- Réaliser une recette fonctionnelle complète sur un poste Windows propre.
- Sécuriser les identifiants par défaut, les droits administrateur et la gestion des données sensibles avant diffusion.
