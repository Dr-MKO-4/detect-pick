"""Fitness BiVAT pour la recherche génétique.

Proxy fitness sans vérité terrain :
  1. Cohérence LOF*–BiVAT : fraction des anomalies LOF* retrouvées par BiVAT.
  2. Score de reconstruction (ELBO proxy) sur les périodes documentées.
  3. Pénalité sur le taux de faux-positifs.

Usage
-----
    fitness_fn = build_bivat_fitness(lof_pipeline, pays="cameroun", volet="Actif")
    search = RechercheHeuristique(ESPACE_BIVAT_DEFAULT, fitness_fn, n_generations=10)
    best = search.fit()
"""
from __future__ import annotations

import numpy as np
import logging

log = logging.getLogger(__name__)

ESPACE_BIVAT_DEFAULT: dict = {
    "d_model": [32, 64, 128, 256],
    "n_heads":  [2, 4, 8],
    "n_layers": [1, 2, 3],
    "window":   [8, 12, 16, 24],
    "beta_kl":  [0.1, 0.5, 1.0, 2.0],
    "lr":       [5e-4, 1e-3, 3e-3],
    "epochs":   [30, 50, 80],
}


def build_bivat_fitness(
    lof_pipeline,
    pays: str  = "cameroun",
    volet: str = "Actif",
    max_anom_frac: float = 0.30,
) -> "callable":
    """
    Fabrique la fonction fitness BiVAT.

    Nécessite que lof_pipeline.fit(pays, volet) ait déjà été appelé.

    Paramètres
    ----------
    lof_pipeline  : instance PipelineLOF déjà entraînée
    pays, volet   : cible
    max_anom_frac : pénalité si taux anomalies BiVAT > ce seuil

    Retourne
    --------
    fitness_fn(params: dict) -> float ∈ [0, 1]
    """
    from bivat import config as C

    def fitness_fn(params: dict) -> float:
        try:
            from bivat.pipeline import PipelineBiVAT
            from bivat.architecture import BiVAT
            from bivat.training import train as bivat_train
            from bivat.scoring import BiVATScorer

            epochs   = int(params.get("epochs", 50))
            d_model  = int(params.get("d_model", C.D_MODEL))
            n_heads  = int(params.get("n_heads", C.NHEAD))
            n_layers = int(params.get("n_layers", C.N_ENCODER_LAYERS))
            window   = int(params.get("window", C.WINDOW_SIZE))
            beta_kl  = float(params.get("beta_kl", C.BETA))
            lr       = float(params.get("lr", C.LR))

            if d_model % n_heads != 0:
                return 0.0

            b = PipelineBiVAT()
            b.fit_from_lof(
                lof_pipeline,
                pays,
                volet,
                epochs        = epochs,
                d_model       = d_model,
                n_heads       = n_heads,
                n_layers      = n_layers,
                window        = window,
                beta_kl       = beta_kl,
                lr            = lr,
                force_retrain = True,
                # La fitness n'exploite jamais shap_results — le sauter fait gagner
                # un DeepExplainer complet (forward+backward) par individu évalué.
                compute_shap  = False,
            )

            key     = (pays.lower(), volet)
            lof_sc  = lof_pipeline.scores_lof.get(key)
            lof_tau = lof_pipeline.tau.get(key)

            if lof_sc is None or lof_tau is None:
                return 0.0

            lof_anom = lof_sc > lof_tau

            if b.anomalies_95 is None or b.dates_test is None:
                return 0.0

            # Aligner les anomalies BiVAT sur les dates LOF*
            import pandas as pd
            bv_series = pd.Series(
                b.anomalies_95.astype(float),
                index=pd.DatetimeIndex(b.dates_test),
            )
            lof_test  = lof_anom.reindex(bv_series.index).fillna(False)

            n_lof_anom = int(lof_test.sum())
            if n_lof_anom == 0:
                coherence = 0.5
            else:
                # Fraction des anomalies LOF* détectées aussi par BiVAT
                coherence = float((bv_series[lof_test] > 0.5).sum()) / n_lof_anom

            n_bv    = len(bv_series)
            bv_frac = float(bv_series.sum()) / max(n_bv, 1)
            fp_pen  = max(0.0, bv_frac - max_anom_frac)

            fitness = 0.70 * coherence - 0.30 * fp_pen

            log.debug("BiVAT fitness  epochs=%d → coherence=%.3f fp_pen=%.3f fitness=%.4f",
                      epochs, coherence, fp_pen, fitness)
            return float(np.clip(fitness, 0.0, 1.0))

        except Exception as exc:
            log.warning("BiVAT fitness erreur : %s", exc)
            return 0.0

    return fitness_fn


if __name__ == "__main__":
    import argparse, sys, json, logging
    sys.path.insert(0, ".")
    from heuristic_search import RechercheHeuristique
    from beac_lof.pipeline import PipelineLOF

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    p = argparse.ArgumentParser(description="Recherche heuristique des hyperparamètres BiVAT")
    p.add_argument("--pays",        default="cameroun")
    p.add_argument("--volet",       default="Actif", choices=["Actif", "Passif"])
    p.add_argument("--data-dir",    default="data")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--population",  type=int, default=8)
    p.add_argument("--patience",    type=int, default=4)
    p.add_argument("--out",         default=None, help="Fichier JSON pour les meilleurs HP")
    args = p.parse_args()

    print(f"Recherche BiVAT  pays={args.pays}  volet={args.volet}  "
          f"gen={args.generations}  pop={args.population}")
    print("  Etape 1/2 : LOF en cours...")

    lof = PipelineLOF(data_dir=args.data_dir)
    lof.fit(args.pays, args.volet)
    print(f"  LOF termine  tau={lof.tau.get((args.pays.lower(), args.volet), '?'):.4f}")

    print("  Etape 2/2 : recherche heuristique BiVAT...")
    fitness_fn = build_bivat_fitness(lof_pipeline=lof, pays=args.pays, volet=args.volet)
    search = RechercheHeuristique(
        espace            = ESPACE_BIVAT_DEFAULT,
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

    print(f"\nMeilleurs hyperparamètres BiVAT ({args.pays}/{args.volet}) :")
    for k, v in best["params"].items():
        print(f"  {k:20s} = {v}")
    print(f"Score fitness : {best['score']:.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print(f"Resultats sauvegardes dans {args.out}")

    sys.exit(0)
