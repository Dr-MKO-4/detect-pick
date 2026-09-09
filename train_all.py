"""
Pré-entraînement BiVAT sur toutes les paires CEMAC (6 pays × 2 volets).

LOF* est non-paramétrique : il s'exécute en direct dans l'application et ne
nécessite aucun pré-entraînement ni sauvegarde de poids.

BiVAT (réseau de neurones) doit être entraîné une fois hors-ligne.
Les poids sont sauvegardés dans :
    models/bivat_{pays}_{volet}.pt

L'application les charge automatiquement sans ré-entraîner.

Usage
-----
    python train_all.py                     # toutes les 12 paires
    python train_all.py --pays cameroun     # un pays, les deux volets
    python train_all.py --pays gabon --volet Actif
    python train_all.py --force             # ré-entraîne même si les poids existent
"""

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PAYS_CEMAC = ["cameroun", "congo", "gabon", "guinee_eq", "rca", "tchad"]
VOLETS     = ["Actif", "Passif"]


def _parse():
    p = argparse.ArgumentParser(description="Pré-entraînement BiVAT BEAC-CEMAC (LOF s'exécute en direct dans l'app)")
    p.add_argument("--pays",     default=None, help="Un seul pays (défaut: tous)")
    p.add_argument("--volet",    default=None, choices=["Actif", "Passif"],
                   help="Un seul volet (défaut: les deux)")
    p.add_argument("--data-dir", default="data", help="Dossier XLSX (défaut: data/)")
    p.add_argument("--force",    action="store_true",
                   help="Ré-entraîner BiVAT même si les poids existent")
    p.add_argument("--epochs",   type=int, default=None,
                   help="Nombre d'époques BiVAT (défaut: valeur config.py)")
    return p.parse_args()


def main():
    args = _parse()

    pays_list = [args.pays]  if args.pays  else PAYS_CEMAC
    volets    = [args.volet] if args.volet else VOLETS
    n_paires  = len(pays_list) * len(volets)

    log.info("=" * 60)
    log.info("Pré-entraînement BiVAT BEAC-CEMAC")
    log.info("  LOF* : non-paramétrique — s'exécute en direct dans l'app (pas de pré-entraînement)")
    log.info("  Paires BiVAT : %d  (%s × %s)", n_paires, pays_list, volets)
    log.info("  Données  : %s/", args.data_dir)
    log.info("=" * 60)

    t0 = time.time()

    # ── 1. LOF sur toutes les paires (pour obtenir les résidus nécessaires à BiVAT) ──
    from beac_lof.pipeline import PipelineLOF

    lof = PipelineLOF(data_dir=args.data_dir)
    log.info("[LOF] fit_all() démarré pour extraire les résidus STL/RPCA…")
    resultats_lof = lof.fit_all(pays_list=pays_list, volets=volets)
    n_ok = len(resultats_lof)
    log.info("[LOF] fit_all() terminé — %d/%d paires réussies", n_ok, n_paires)

    if not resultats_lof:
        log.error("Aucun fichier traité. Vérifiez --data-dir (%s) et les fichiers XLSX.", args.data_dir)
        sys.exit(1)

    # Tableau récapitulatif LOF
    print()
    print(f"{'Pays':15} {'Volet':8} {'N obs':>6} {'Indicateurs':>12} {'Anomalies':>10} {'tau':>8}")
    print("-" * 65)
    for (pays, volet), r in resultats_lof.items():
        print(f"{pays.capitalize():15} {volet:8} {r['n_obs']:>6} "
              f"{r['n_indicateurs']:>12} {r['n_anomalies_tau']:>10} {r['tau']:>8.4f}")
    print()

    # ── 2. BiVAT sur chaque paire LOF réussie ────────────────────────────────
    from bivat.pipeline import PipelineBiVAT
    from bivat import config as C

    epochs    = args.epochs or C.EPOCHS
    n_bivat_ok = 0

    for pays in pays_list:
        for volet in volets:
            k = (pays, volet)
            if k not in resultats_lof:
                log.warning("[BIVAT] Paire non disponible (LOF échoué) : %s/%s", pays, volet)
                continue

            weights = os.path.join("models", f"bivat_{pays}_{volet.lower()}.pt")
            if not args.force and os.path.exists(weights):
                log.info("[BIVAT] %s/%s — poids existants (%s), ignoré (--force pour ré-entraîner)",
                         pays, volet, weights)
                n_bivat_ok += 1
                continue

            log.info("[BIVAT] Entraînement %s/%s  (%d époques max)…", pays, volet, epochs)
            t1 = time.time()
            try:
                b = PipelineBiVAT()
                b.fit_from_lof(lof, pays, volet, epochs=epochs, force_retrain=args.force)
                q95 = b.q_hat_95
                n_anom = int(b.anomalies_95.sum()) if b.anomalies_95 is not None else "?"
                log.info(
                    "[BIVAT] %s/%s terminé en %.1fs · q̂₉₅=%.4f · %s anomalies (test)",
                    pays, volet, time.time() - t1, q95 or 0, n_anom,
                )
                n_bivat_ok += 1
            except Exception as exc:
                log.error("[BIVAT] Erreur %s/%s : %s", pays, volet, exc, exc_info=True)

    # ── Résumé final ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    log.info("=" * 60)
    log.info("Résumé final")
    log.info("  LOF   : %d/%d paires", n_ok, n_paires)
    log.info("  BiVAT : %d/%d paires", n_bivat_ok, n_ok)
    log.info("  Durée : %.1f s (%.1f min)", elapsed, elapsed / 60)
    log.info("  Poids sauvegardés dans : models/bivat_{pays}_{volet}.pt")
    log.info("=" * 60)

    sys.exit(0 if n_bivat_ok == n_ok else 1)


if __name__ == "__main__":
    main()
