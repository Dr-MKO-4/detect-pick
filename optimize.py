"""CLI unifiée de recherche heuristique des hyperparamètres (LOF* / BiVAT).

Usage
-----
    python optimize.py --modele lof   --pays cameroun --volet Actif
    python optimize.py --modele bivat --pays cameroun --volet Actif

Résultats sauvegardés par défaut dans models/hp/{pays}_{volet}_{modele}.json
(voir LANCEMENT.md, section 1c).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable

# Rend le script exécutable depuis n'importe quel répertoire courant,
# pas seulement la racine du projet.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from heuristic_search import RechercheHeuristique
from optimization.lof_search import ESPACE_LOF_DEFAULT, build_lof_fitness
from optimization.bivat_search import ESPACE_BIVAT_DEFAULT, build_bivat_fitness

log = logging.getLogger(__name__)

# Defauts par modele : (generations, population, patience).
DEFAULTS = {
    "lof":   {"generations": 15, "population": 12, "patience": 5},
    "bivat": {"generations": 10, "population": 8,  "patience": 4},
}


def _run_search(espace: dict, fitness_fn: Callable, args: argparse.Namespace) -> dict:
    search = RechercheHeuristique(
        espace            = espace,
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
    return search.fit()


def _optimiser_lof(args: argparse.Namespace) -> dict:
    fitness_fn = build_lof_fitness(data_dir=args.data_dir, pays=args.pays, volet=args.volet)
    return _run_search(ESPACE_LOF_DEFAULT, fitness_fn, args)


def _optimiser_bivat(args: argparse.Namespace) -> dict:
    from beac_lof.pipeline import PipelineLOF

    print("  Etape 1/2 : LOF en cours...")
    lof = PipelineLOF(data_dir=args.data_dir)
    lof.fit(args.pays, args.volet)
    tau = lof.tau.get((args.pays.lower(), args.volet))
    print(f"  LOF termine  tau={tau:.4f}" if tau is not None else "  LOF termine  tau=?")

    print("  Etape 2/2 : recherche heuristique BiVAT...")
    fitness_fn = build_bivat_fitness(lof_pipeline=lof, pays=args.pays, volet=args.volet)
    return _run_search(ESPACE_BIVAT_DEFAULT, fitness_fn, args)


OPTIMISEURS: dict[str, Callable[[argparse.Namespace], dict]] = {
    "lof":   _optimiser_lof,
    "bivat": _optimiser_bivat,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recherche heuristique des hyperparamètres (LOF / BiVAT)")
    p.add_argument("--modele",      required=True, choices=list(OPTIMISEURS))
    p.add_argument("--pays",        default="cameroun")
    p.add_argument("--volet",       default="Actif", choices=["Actif", "Passif"])
    p.add_argument("--data-dir",    default="data")
    p.add_argument("--generations", type=int, default=None,
                    help="Defaut : 15 (lof) / 10 (bivat)")
    p.add_argument("--population",  type=int, default=None,
                    help="Defaut : 12 (lof) / 8 (bivat)")
    p.add_argument("--patience",    type=int, default=None,
                    help="Defaut : 5 (lof) / 4 (bivat)")
    p.add_argument("--out",         default=None,
                    help="Fichier JSON de sortie (defaut : models/hp/{pays}_{volet}_{modele}.json)")
    args = p.parse_args(argv)

    defaults = DEFAULTS[args.modele]
    args.generations = args.generations or defaults["generations"]
    args.population  = args.population or defaults["population"]
    args.patience    = args.patience or defaults["patience"]
    return args


def main(argv: list[str] | None = None) -> None:
    # Console Windows (cp1252) : évite un crash sur les caractères non-ASCII
    # affichés par heuristic_search.py (ex. 'μ').
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    args = parse_args(argv)

    print(f"Recherche {args.modele.upper()}  pays={args.pays}  volet={args.volet}  "
          f"gen={args.generations}  pop={args.population}")

    best = OPTIMISEURS[args.modele](args)

    print(f"\nMeilleurs hyperparamètres {args.modele.upper()} ({args.pays}/{args.volet}) :")
    for k, v in best["params"].items():
        print(f"  {k:20s} = {v}")
    print(f"Score fitness : {best['score']:.4f}")

    out_path = Path(args.out) if args.out else Path("models/hp") / f"{args.pays}_{args.volet}_{args.modele}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Resultats sauvegardes dans {out_path}")


if __name__ == "__main__":
    main()
