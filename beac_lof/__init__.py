"""
beac_lof  Package de détection d'anomalies LOF sur les données monétaires BEAC.

Structure
---------
  config        : constantes et paramètres méthodologiques
  loader        : chargement et parsing des fichiers XLSX (§2.1)
  preprocessing : imputation et standardisation MAD (§2.2–2.3)
  decomposition : décomposition STL (§3)
  reduction     : RPCA + ACP (§4)
  detection     : LOF* et seuil τ (§5/6)
  pipeline      : orchestration  classe PipelineLOF
  graphiques    : 14 figures Plotly  classe BEACGraphiques

Utilisation rapide
------------------
    from beac_lof import PipelineLOF

    pipeline = PipelineLOF(data_dir="data")
    pipeline.fit_all()

    fig = pipeline.vers_graphiques().fig10_serie_temporelle_lof("cameroun", "Actif")
    fig.show()

Intégration FastAPI (future)
----------------------------
    from beac_lof import PipelineLOF
    from beac_lof.detection import lof_star, seuil_tau
    from beac_lof.loader import charger_fichier
"""

from .pipeline import PipelineLOF
from .graphiques import BEACGraphiques

__all__ = ["PipelineLOF", "BEACGraphiques"]
__version__ = "0.1.0"
__author__ = "MOUKOKO EKAMBI GEORGES MIGUEL"
