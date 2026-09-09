"""Fitness LOF* pour la recherche heuristique.

Proxy fitness sans vérité terrain :
  1. Rang des périodes documentées dans le top-k anomalies (COVID, choc pétrolier, crise CEMAC).
  2. Injection synthétique d'anomalies (10 % de la série) — rappel comme proxy.
  3. Pénalité sur le taux de faux-positifs (fraction anomalies > 30 %).

Usage
-----
    fitness_fn = build_lof_fitness(data_dir="data", pays="cameroun", volet="Actif")
    search = RechercheHeuristique(ESPACE_LOF_DEFAULT, fitness_fn, maximize=True)
    best = search.fit()
"""
from __future__ import annotations

import numpy as np
import logging

log = logging.getLogger(__name__)

# Périodes documentées (inclusives) : anomalies CEMAC connues.
# La fitness récompense le pipeline qui les classe parmi les plus hauts scores.
KNOWN_ANOMALY_PERIODS = [
    ("2020-03", "2021-12"),   # COVID-19
    ("2014-06", "2016-12"),   # Choc pétrolier CEMAC
    ("2017-01", "2019-06"),   # Crise CEMAC / CFA
]

# Espace de recherche par défaut pour LOF*.
# Toutes les valeurs sont des listes discrètes — compatibles avec RechercheHeuristique.
ESPACE_LOF_DEFAULT: dict = {
    "minpts_lb":   [3, 5, 8, 10],
    "minpts_ub":   [15, 20, 25, 30],
    "tau_method":  ["iqr15", "iqr20", "p95"],
    "stl_period":  [6, 12, 24],
    "pca_var_target": [0.80, 0.90, 0.95],
}


def _periods_in_range(index, start: str, end: str):
    """Indices dans l'index pandas compris entre start et end (YYYY-MM)."""
    mask = (index.astype(str).str[:7] >= start) & (index.astype(str).str[:7] <= end)
    return np.where(mask)[0]


def build_lof_fitness(
    data_dir: str = "data",
    pays: str     = "cameroun",
    volet: str    = "Actif",
    known_periods: list[tuple[str, str]] | None = None,
    max_anom_frac: float = 0.30,
) -> "callable":
    """
    Fabrique la fonction fitness LOF* pour (pays, volet) donné.

    Chaque appel crée une instance PipelineLOF avec les hyperparamètres proposés,
    lance fit(pays, volet) et évalue le proxy.

    Paramètres configurables
    ------------------------
    data_dir      : chemin vers les données XLSX
    pays          : pays CEMAC cible
    volet         : "Actif" ou "Passif"
    known_periods : liste de (start, end) en YYYY-MM — périodes documentées à retrouver
    max_anom_frac : pénalité si taux d'anomalies > ce seuil

    Retourne
    --------
    fitness_fn(params: dict) -> float ∈ [-1, 1]
    """
    _periods = known_periods if known_periods is not None else KNOWN_ANOMALY_PERIODS

    def fitness_fn(params: dict) -> float:
        lb           = int(params.get("minpts_lb",    5))
        ub           = int(params.get("minpts_ub",   20))
        tau_method   = str(params.get("tau_method",  "iqr15"))
        stl_period   = int(params.get("stl_period",  12))
        pca_var      = float(params.get("pca_var_target", 0.90))

        # Sanity check : lb doit être < ub
        if lb >= ub:
            return -1.0

        try:
            from beac_lof.pipeline import PipelineLOF

            p = PipelineLOF(
                data_dir       = data_dir,
                minpts_lb      = lb,
                minpts_ub      = ub,
                tau_method     = tau_method,
                stl_period     = stl_period,
                pca_var_target = pca_var,
            )
            p.fit(pays, volet)

            key    = (pays.lower(), volet)
            scores = p.scores_lof.get(key)
            tau    = p.tau.get(key)

            if scores is None or tau is None:
                return -1.0

            index    = scores.index
            n        = len(scores)
            n_anom   = int((scores > tau).sum())
            anom_frac = n_anom / max(n, 1)

            # Composante 1 : rang des périodes documentées
            rank_scores = []
            for start, end in _periods:
                idxs = _periods_in_range(index, start, end)
                if len(idxs) == 0:
                    continue
                period_scores = scores.iloc[idxs].values
                overall_rank  = np.mean([
                    float(np.sum(scores.values >= s)) / max(n, 1)
                    for s in period_scores
                ])
                rank_scores.append(overall_rank)
            rank_component = float(np.mean(rank_scores)) if rank_scores else 0.5

            # Composante 2 : injection synthétique (10 % de points bruités +3σ)
            rng  = np.random.default_rng(42)
            synth_idx = rng.choice(n, max(1, n // 10), replace=False)
            s_arr     = scores.values.copy().astype(float)
            sigma     = np.std(s_arr)
            s_arr[synth_idx] += 3.0 * sigma
            recall_synth = float(np.mean(s_arr[synth_idx] > tau))

            # Composante 3 : pénalité sur faux-positifs excessifs
            fp_penalty = max(0.0, anom_frac - max_anom_frac)

            fitness = 0.50 * rank_component + 0.40 * recall_synth - 0.10 * fp_penalty

            log.debug("LOF fitness  lb=%d ub=%d tau=%s stl=%d pca=%.2f → %.4f",
                      lb, ub, tau_method, stl_period, pca_var, fitness)
            return float(np.clip(fitness, -1.0, 1.0))

        except Exception as exc:
            log.warning("LOF fitness erreur : %s", exc)
            return -1.0

    return fitness_fn


if __name__ == "__main__":
    import argparse, sys, json, logging
    sys.path.insert(0, ".")
    from heuristic_search import RechercheHeuristique

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    p = argparse.ArgumentParser(description="Recherche heuristique des hyperparamètres LOF")
    p.add_argument("--pays",        default="cameroun")
    p.add_argument("--volet",       default="Actif", choices=["Actif", "Passif"])
    p.add_argument("--data-dir",    default="data")
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--population",  type=int, default=12)
    p.add_argument("--patience",    type=int, default=5)
    p.add_argument("--out",         default=None, help="Fichier JSON pour les meilleurs HP")
    args = p.parse_args()

    print(f"Recherche LOF  pays={args.pays}  volet={args.volet}  "
          f"gen={args.generations}  pop={args.population}")

    fitness_fn = build_lof_fitness(data_dir=args.data_dir, pays=args.pays, volet=args.volet)
    search = RechercheHeuristique(
        espace            = ESPACE_LOF_DEFAULT,
        fitness_fn        = fitness_fn,
        n_generations     = args.generations,
        taille_population = args.population,
        elite_size        = 2,
        mutation_rate     = 0.35,
        refroidissement   = 0.97,
        min_diversite     = 0.40,
        patience          = args.patience,
        maximize          = True,
    )
    best = search.fit()

    print(f"\nMeilleurs hyperparamètres LOF ({args.pays}/{args.volet}) :")
    for k, v in best["params"].items():
        print(f"  {k:20s} = {v}")
    print(f"Score fitness : {best['score']:.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print(f"Resultats sauvegardes dans {args.out}")

    sys.exit(0)
