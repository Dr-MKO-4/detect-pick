"""
beac_lof/pipeline.py  Orchestration du pipeline LOF-BEAC (§1–§7).

La classe PipelineLOF enchaîne les modules du package :
  loader → preprocessing → decomposition → reduction → detection

Utilisation
-----------
    from beac_lof import PipelineLOF

    pipeline = PipelineLOF()
    pipeline.fit("cameroun", "Actif")   # un fichier
    pipeline.fit_all()                   # 12 fichiers

    g = pipeline.vers_graphiques()       # → BEACGraphiques
    fig = g.fig10_serie_temporelle_lof("cameroun", "Actif")
    fig.show()

Pour FastAPI (usage futur) :
    Les dictionnaires d'attributs exposent les données intermédiaires comme
    structures Python/numpy/pandas directement sérialisables vers JSON.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR, PAYS_CEMAC, VOLETS, FILE_MAP,
    MINPTS_LB, MINPTS_UB, PCA_VAR_TARGET,
    SYSTEMIC_M_MIN,
)
from .loader import charger_fichier
from .preprocessing import imputer, mad_standardize
from .decomposition import stl_decompose
from .reduction import rpca_ialm, pca_sur_L
from .detection import lof_star, seuil_tau

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Cache module-level des étapes de pipeline indépendantes de (minpts_lb, minpts_ub,
# tau_method) : chargement, imputation, standardisation, STL, RPCA, ACP.
# Utilisé par la recherche heuristique (optimization/lof_search.py) qui instancie
# un nouveau PipelineLOF à chaque évaluation de fitness — sans ce cache, ces étapes
# coûteuses (MICE, RPCA-IALM) seraient ré-exécutées à l'identique pour chaque
# combinaison de minpts_lb/ub/tau_method testée.
_UPSTREAM_CACHE: dict = {}


def _upstream_cache_key(
    data_dir: Path, pays: str, volet: str,
    pca_var_target: float, rpca_lam, rpca_tol: float, rpca_max_iter: int,
    stl_period,
) -> tuple:
    fname = None
    try:
        from .config import FILE_MAP
        fname = f"{FILE_MAP[pays.lower()]}_{volet}.xlsx"
        mtime = (data_dir / fname).stat().st_mtime
    except Exception:
        mtime = None
    return (
        str(data_dir.resolve()), pays.lower(), volet, fname, mtime,
        pca_var_target, rpca_lam, rpca_tol, rpca_max_iter, stl_period,
    )


class PipelineLOF:
    """
    Pipeline complet de détection d'anomalies LOF sur les données monétaires BEAC.

    Attributs publics après fit()
    -----------------------------
    data_brute         : dict[(pays, volet) → pd.DataFrame]    données avant imputation
    data_imputed       : dict[(pays, volet) → pd.DataFrame]    après imputation
    data_std           : dict[(pays, volet) → pd.DataFrame]    après MAD
    residus            : dict[(pays, volet) → pd.DataFrame]    résidus STL R_{i,t}
    data_stl           : dict[(pays, volet) → dict]            composantes STL complètes
    L_rpca             : dict[(pays, volet) → np.ndarray]      composante L de RPCA
    S_rpca             : dict[(pays, volet) → np.ndarray]      composante S de RPCA
    composantes_pca    : dict[(pays, volet) → np.ndarray]      projections ACP (n × k)
    loadings_pca       : dict[(pays, volet) → pd.DataFrame]    vecteurs propres (k × p)
    variance_explained : dict[(pays, volet) → np.ndarray]      variance par composante
    lof_par_minpts     : dict[(pays, volet) → pd.DataFrame]    LOF(MinPts), colonnes=MinPts
    scores_lof         : dict[(pays, volet) → pd.Series]       LOF* (index = DatetimeIndex)
    tau                : dict[(pays, volet) → float]            seuil τ IQR
    tau_p95            : dict[(pays, volet) → float]            seuil p95 (variante)
    indicateurs_exclus : dict[(pays, volet) → list]             IFS codes exclus (MNAR)
    embedding_umap     : dict[(pays, volet) → np.ndarray]      UMAP 2-D (si installé)
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        minpts_lb: int = MINPTS_LB,
        minpts_ub: int = MINPTS_UB,
        pca_var_target: float = PCA_VAR_TARGET,
        rpca_lam: Optional[float] = None,
        rpca_tol: float = 1e-6,
        rpca_max_iter: int = 150,
        stl_period: Optional[int] = None,
        tau_method: str = "iqr15",
    ):
        self.data_dir       = Path(data_dir)
        self.minpts_lb      = minpts_lb
        self.minpts_ub      = minpts_ub
        self.pca_var_target = pca_var_target
        self.rpca_lam       = rpca_lam
        self.rpca_tol       = rpca_tol
        self.rpca_max_iter  = rpca_max_iter
        self.stl_period     = stl_period
        self.tau_method     = tau_method

        # Résultats intermédiaires
        self.data_brute:         dict = {}
        self.data_imputed:       dict = {}
        self.data_std:           dict = {}
        self.residus:            dict = {}
        self.data_stl:           dict = {}
        self.L_rpca:             dict = {}
        self.S_rpca:             dict = {}
        self.composantes_pca:    dict = {}
        self.loadings_pca:       dict = {}
        self.variance_explained: dict = {}
        self.lof_par_minpts:     dict = {}
        self.scores_lof:         dict = {}
        self.tau:                dict = {}
        self.tau_p95:            dict = {}
        self.indicateurs_exclus: dict = {}
        self.embedding_umap:     dict = {}

    # ── Pipeline pour un fichier ──────────────────────────────────────────────

    def _compute_upstream(self, pays: str, volet: str) -> dict:
        """
        Exécute les étapes du pipeline indépendantes de (minpts_lb, minpts_ub,
        tau_method) : chargement → imputation → MAD → STL → RPCA → ACP.

        Résultat mis en cache (module-level) par (data_dir, pays, volet, stl_period,
        pca_var_target, paramètres RPCA) — ces étapes sont les plus coûteuses du
        pipeline (MICE PyTorch, RPCA-IALM) et ne dépendent pas des hyperparamètres
        LOF* testés par la recherche heuristique.
        """
        key = _upstream_cache_key(
            self.data_dir, pays, volet,
            self.pca_var_target, self.rpca_lam, self.rpca_tol, self.rpca_max_iter,
            self.stl_period,
        )
        cached = _UPSTREAM_CACHE.get(key)
        if cached is not None:
            log.info("  (cache) étapes amont réutilisées pour %s/%s", pays, volet)
            return cached

        # §2.1  Chargement + troncature structurelle
        df_brut = charger_fichier(self.data_dir, pays, volet)
        log.info("  Chargé : %d mois × %d indicateurs", *df_brut.shape)

        # §2.2  Imputation (interpolation / MICE / exclusion MNAR)
        df_imp, excluded = imputer(df_brut)
        log.info("  Imputé : %d indicateurs (%d exclus)", df_imp.shape[1], len(excluded))

        # §2.3  Standardisation MAD
        df_std = mad_standardize(df_imp)

        # §3  Décomposition STL
        residus, composantes_stl = stl_decompose(df_std, period=self.stl_period)
        log.info("  STL : résidus %s", residus.shape)

        # §4.1  RPCA
        R = residus.values.astype(float)
        L, S = rpca_ialm(R, lam=self.rpca_lam, tol=self.rpca_tol, max_iter=self.rpca_max_iter)
        log.info("  RPCA : rang(L)=%d, densité(S)=%.1f%%",
                 np.linalg.matrix_rank(L), (np.abs(S) > 1e-4).mean() * 100)

        # §4.2  ACP sur L
        proj, loadings_df, var_ratio = pca_sur_L(L, residus.columns, self.pca_var_target)
        log.info("  ACP : %d composantes (%.1f%% variance)",
                 proj.shape[1], var_ratio[:proj.shape[1]].sum() * 100)

        result = {
            "df_brut": df_brut, "df_imp": df_imp, "excluded": excluded,
            "df_std": df_std,
            "residus": residus, "composantes_stl": composantes_stl,
            "L": L, "S": S,
            "proj": proj, "loadings_df": loadings_df, "var_ratio": var_ratio,
        }
        _UPSTREAM_CACHE[key] = result
        return result

    def fit(self, pays: str, volet: str) -> dict:
        """
        Exécute le pipeline complet sur le fichier (pays, volet).

        Retourne un dict synthétique avec les métriques clés.
        """
        k = (pays.lower(), volet)
        log.info("=== %s / %s ===", pays.capitalize(), volet)

        up = self._compute_upstream(pays, volet)
        self.data_brute[k]          = up["df_brut"].copy()
        self.data_imputed[k]        = up["df_imp"]
        self.indicateurs_exclus[k]  = up["excluded"]
        residus                     = up["residus"]
        self.residus[k]             = residus
        self.data_std[k]            = up["df_std"]
        self.data_stl[k]            = up["composantes_stl"]
        self.L_rpca[k]              = up["L"]
        self.S_rpca[k]              = up["S"]
        proj                        = up["proj"]
        self.composantes_pca[k]     = proj
        self.loadings_pca[k]        = up["loadings_df"]
        self.variance_explained[k]  = up["var_ratio"]

        # §5/6  LOF*
        n_obs = proj.shape[0]
        df_mp, scores = lof_star(proj, n_obs, self.minpts_lb, self.minpts_ub)
        df_mp.index = residus.index
        self.lof_par_minpts[k] = df_mp
        self.scores_lof[k]     = pd.Series(scores, index=residus.index, name="LOF*")

        # §6  Seuil τ
        tau_val, p95_val = seuil_tau(scores, method=self.tau_method)
        self.tau[k]     = tau_val
        self.tau_p95[k] = p95_val

        n_anom = int((scores > tau_val).sum())
        log.info("  τ=%.3f → %d anomalies | p95=%.3f", tau_val, n_anom, p95_val)

        # §4  UMAP (visualisation uniquement, non déterministe)
        self._fit_umap(k, proj)

        return {
            "pays": pays, "volet": volet,
            "n_obs": n_obs,
            "n_indicateurs": up["df_imp"].shape[1],
            "n_anomalies_tau": n_anom,
            "tau": tau_val,
            "tau_p95": p95_val,
        }

    def _fit_umap(self, k: tuple, proj: np.ndarray):
        """UMAP optionnel (§4  visualisation, non utilisé pour le calcul LOF)."""
        try:
            import umap
            self.embedding_umap[k] = umap.UMAP(
                n_components=2, random_state=42, n_neighbors=15
            ).fit_transform(proj)
        except ImportError:
            pass

    # ── Pipeline multi-fichiers ───────────────────────────────────────────────

    def fit_all(
        self,
        pays_list: Optional[list] = None,
        volets: Optional[list] = None,
    ) -> dict:
        """
        Exécute le pipeline sur tous les fichiers disponibles (mode séparé §1.2).

        Les fichiers manquants sont ignorés avec un avertissement.
        """
        pays_list = pays_list or PAYS_CEMAC
        volets    = volets or VOLETS
        resultats = {}
        for pays in pays_list:
            for volet in volets:
                fname = f"{FILE_MAP[pays]}_{volet}.xlsx"
                if not (self.data_dir / fname).exists():
                    log.warning("Fichier manquant, ignoré : %s", fname)
                    continue
                try:
                    resultats[(pays, volet)] = self.fit(pays, volet)
                except Exception as exc:
                    log.error("Erreur %s/%s : %s", pays, volet, exc)
        return resultats

    # ── Validation (§7) ──────────────────────────────────────────────────────

    def anomalies_systemiques(
        self, volet: str = "Actif", m_min: int = SYSTEMIC_M_MIN
    ) -> pd.Series:
        """
        Mois anomaux dans ≥ m_min pays simultanément (§7.4  anomalies systémiques CEMAC).

        Retourne
        --------
        pd.Series  nombre de pays en anomalie par mois (index = DatetimeIndex).
        """
        series = []
        for pays in PAYS_CEMAC:
            k = (pays, volet)
            if k not in self.scores_lof or k not in self.tau:
                continue
            series.append((self.scores_lof[k] > self.tau[k]).rename(pays))
        if not series:
            return pd.Series(dtype=int)
        mat   = pd.concat(series, axis=1).infer_objects(copy=False).fillna(False).astype(int)
        count = mat.sum(axis=1)
        return count[count >= m_min]

    def coherence_actif_passif(self, pays: str) -> pd.DataFrame:
        """
        Cohérence Actif–Passif mois par mois (§7.5).

        Retourne un DataFrame avec :
        lof_actif, lof_passif, anom_actif, anom_passif, concordant, discordant.
        """
        ka = (pays.lower(), "Actif")
        kp = (pays.lower(), "Passif")
        sa = self.scores_lof.get(ka, pd.Series(dtype=float))
        sp = self.scores_lof.get(kp, pd.Series(dtype=float))
        if sa.empty or sp.empty:
            return pd.DataFrame()
        df = pd.DataFrame({"lof_actif": sa, "lof_passif": sp}).dropna()
        df["anom_actif"]  = df["lof_actif"]  > self.tau.get(ka, np.inf)
        df["anom_passif"] = df["lof_passif"] > self.tau.get(kp, np.inf)
        df["concordant"]  = df["anom_actif"] & df["anom_passif"]
        df["discordant"]  = df["anom_actif"] ^ df["anom_passif"]
        return df

    def top_anomalies(self, pays: str, volet: str, n: int = 10) -> pd.DataFrame:
        """
        Tableau des n mois les plus anomaux (§7.6  présentation des résultats).

        Colonnes : LOF*, indicateurs_dominants (3 premiers par |résidu|).
        """
        k  = (pays.lower(), volet)
        sc = self.scores_lof.get(k, pd.Series(dtype=float))
        res = self.residus.get(k, pd.DataFrame())
        if sc.empty:
            return pd.DataFrame()
        rows = []
        for dt, val in sc.nlargest(n).items():
            ind_dom = (res.loc[dt].abs().nlargest(3).index.tolist()
                       if dt in res.index else [])
            rows.append({
                "date": dt,
                "LOF*": round(val, 4),
                "indicateurs_dominants": " | ".join(ind_dom),
            })
        return pd.DataFrame(rows).set_index("date")

    # ── Intégration graphiques ────────────────────────────────────────────────

    def vers_graphiques(self):
        """
        Crée un BEACGraphiques pré-chargé avec tous les résultats du pipeline.

        Exemple
        -------
            g = pipeline.vers_graphiques()
            g.fig10_serie_temporelle_lof("cameroun", "Actif").show()
        """
        from .graphiques import BEACGraphiques
        return BEACGraphiques(
            residus=self.residus,
            scores_lof=self.scores_lof,
            data_brute=self.data_brute,
            tau=self.tau,
            composantes_pca=self.composantes_pca,
            loadings_pca=self.loadings_pca,
            variance_explained=self.variance_explained,
            embedding_umap=self.embedding_umap,
            lof_par_minpts=self.lof_par_minpts,
            data_stl=self.data_stl,
        )

    def get_results(self) -> dict:
        """
        Serialisable results dict for Dash dcc.Store.
        Returns scores, thresholds and metadata for all fitted (pays, volet) pairs.
        """
        out: dict = {}
        for key, sc in self.scores_lof.items():
            pays, volet = key
            out[f"{pays}_{volet}"] = {
                "scores":  sc.tolist(),
                "dates":   [str(d)[:10] for d in sc.index],
                "tau":     self.tau.get(key),
                "tau_p95": self.tau_p95.get(key),
                "n_anom":  int((sc > self.tau[key]).sum()) if key in self.tau else None,
            }
        return out

    def generer_rapport_complet(
        self,
        dossier_export: str = "rapport",
        format: str = "html",
        pays_list: Optional[list] = None,
        volets: Optional[list] = None,
    ) -> dict:
        """
        Génère les 14 figures pour chaque (pays, volet) traité et les exporte.

        Structure de sortie :
            rapport/<pays>/<volet>/fig{1..14}_*.html
            rapport/fig14_interpays_<volet>.html

        format : "html" (interactif, défaut) ou "png" (kaleido requis).
        """
        import os
        g = self.vers_graphiques()
        pays_list = pays_list or PAYS_CEMAC
        volets    = volets or VOLETS
        resultats = {}

        for pays in pays_list:
            for volet in volets:
                k = (pays.lower(), volet)
                if k not in self.scores_lof:
                    continue
                sous_dossier = os.path.join(dossier_export, pays, volet)
                os.makedirs(sous_dossier, exist_ok=True)
                log.info("Figures  %s/%s → %s", pays, volet, sous_dossier)
                try:
                    resultats[k] = g.generer_toutes(
                        pays=pays, volet=volet,
                        dossier_export=sous_dossier, format=format,
                    )
                except Exception as exc:
                    log.error("  Erreur figures %s/%s : %s", pays, volet, exc)

        # Fig 14 inter-pays (une par volet, agrège tous les pays)
        for volet in volets:
            path = os.path.join(dossier_export, f"fig14_interpays_{volet}.{format}")
            try:
                fig14 = g.fig14_heatmap_interpays(volet=volet, animate=False)
                if format == "html":
                    fig14.write_html(path, include_plotlyjs="cdn")
                else:
                    fig14.write_image(path, scale=2)
                log.info("Fig14 %s → %s", volet, path)
            except Exception as exc:
                log.error("  Erreur Fig14 %s : %s", volet, exc)

        log.info("Rapport complet → '%s'", dossier_export)
        return resultats


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Pipeline LOF-BEAC standalone")
    parser.add_argument("--pays",  default=None, help="Pays (ex: cameroun). Défaut: tous.")
    parser.add_argument("--volet", default=None, choices=["Actif", "Passif"],
                        help="Volet. Défaut: les deux.")
    parser.add_argument("--data-dir", default="data", help="Dossier XLSX (défaut: data/)")
    parser.add_argument("--rapport", action="store_true",
                        help="Exporter les 14 figures HTML dans rapport/")
    args = parser.parse_args()

    p = PipelineLOF(data_dir=args.data_dir)

    pays_list  = [args.pays]  if args.pays  else None
    volets     = [args.volet] if args.volet else None

    resultats = p.fit_all(pays_list=pays_list, volets=volets)

    print(f"\n{'Pays':15} {'Volet':8} {'N obs':>6} {'Indicateurs':>12} {'Anomalies':>10} {'tau':>8}")
    print("-" * 65)
    for (pays, volet), r in resultats.items():
        print(f"{pays.capitalize():15} {volet:8} {r['n_obs']:>6} "
              f"{r['n_indicateurs']:>12} {r['n_anomalies_tau']:>10} {r['tau']:>8.4f}")

    if resultats and args.rapport:
        p.generer_rapport_complet(dossier_export="rapport", format="html",
                                  pays_list=pays_list, volets=volets)
        print("\nFigures exportées dans rapport/")

    sys.exit(0 if resultats else 1)
