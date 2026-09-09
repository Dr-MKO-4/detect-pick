"""
beac_lof/graphiques.py  Classe BEACGraphiques
Implémentation des 14 figures interactives/animées du pipeline LOF-BEAC.
Toutes les figures sont produites avec Plotly (go.Figure).

Usage minimal :
    g = BEACGraphiques(residus, scores_lof, data_brute, tau, ...)
    fig = g.fig8_distribution_lof("cameroun", "Actif")
    fig.show()                          # ouvre dans le navigateur
    fig.write_html("fig8.html")         # export HTML standalone
    fig.write_image("fig8.png")         # export PNG (kaleido requis)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

PAYS_CEMAC = ["cameroun", "congo", "gabon", "guinee_eq", "rca", "tchad"]
VOLETS = ["Actif", "Passif"]

_MOIS_FR = ["", "Janv", "Févr", "Mars", "Avr", "Mai", "Juin",
             "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]
_MOIS_LABELS = _MOIS_FR[1:]   # ["Janv", …, "Déc"]


def _fmt_mois(dt) -> str:
    """Formate une date en 'Janv 2015'."""
    try:
        return f"{_MOIS_FR[dt.month]} {dt.year}"
    except Exception:
        return str(dt)[:7]


class BEACGraphiques:
    """
    Conteneur de toutes les figures Plotly du pipeline de détection LOF-BEAC.

    Chaque méthode fig{N}_* retourne un go.Figure directement utilisable.
    La sélection pays/volet est intégrée en dropdown dans la figure.

    Paramètres
    ----------
    residus : dict[(pays, volet) -> pd.DataFrame]
        Résidus STL (colonnes = indicateurs, index = DatetimeIndex mensuel).
    scores_lof : dict[(pays, volet) -> pd.Series]
        Scores LOF* finaux (index = DatetimeIndex mensuel).
    data_brute : dict[(pays, volet) -> pd.DataFrame]
        Données brutes avant imputation (NaN / #VALUE! préservés).
    tau : dict[(pays, volet) -> float]
        Seuil IQR τ = Q3 + 1.5·IQR calculé par fichier.
    composantes_pca : dict[(pays, volet) -> np.ndarray], optionnel
        Projections PCA (n_obs × n_comp).
    loadings_pca : dict[(pays, volet) -> pd.DataFrame], optionnel
        Vecteurs propres (index = n_comp entiers, colonnes = indicateurs).
    variance_explained : dict[(pays, volet) -> np.ndarray], optionnel
        Part de variance expliquée par composante.
    embedding_umap : dict[(pays, volet) -> np.ndarray], optionnel
        Embedding UMAP 2-D (n_obs × 2).
    lof_par_minpts : dict[(pays, volet) -> pd.DataFrame], optionnel
        LOF(MinPts)  colonnes = valeurs entières de MinPts ∈ [10, 30].
    data_stl : dict[(pays, volet) -> dict[str -> pd.DataFrame]], optionnel
        Composantes STL par indicateur : clés "observed", "trend", "seasonal", "residual".
    """

    def __init__(
        self,
        residus: dict,
        scores_lof: dict,
        data_brute: dict,
        tau: dict,
        composantes_pca: Optional[dict] = None,
        loadings_pca: Optional[dict] = None,
        variance_explained: Optional[dict] = None,
        embedding_umap: Optional[dict] = None,
        lof_par_minpts: Optional[dict] = None,
        data_stl: Optional[dict] = None,
    ):
        self.residus = residus
        self.scores_lof = scores_lof
        self.data_brute = data_brute
        self.tau = tau
        self.composantes_pca = composantes_pca or {}
        self.loadings_pca = loadings_pca or {}
        self.variance_explained = variance_explained or {}
        self.embedding_umap = embedding_umap or {}
        self.lof_par_minpts = lof_par_minpts or {}
        self.data_stl = data_stl or {}

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _k(self, pays: str, volet: str) -> tuple:
        return (pays.lower(), volet)

    def _label(self, pays: str, volet: str) -> str:
        return f"{pays.capitalize()} / {volet}"

    def _all_keys(self):
        """Retourne toutes les paires (pays, volet) disponibles dans scores_lof."""
        keys = []
        for pays in PAYS_CEMAC:
            for volet in VOLETS:
                k = self._k(pays, volet)
                if k in self.scores_lof or k in self.residus or k in self.data_brute:
                    keys.append((pays, volet))
        return keys

    def _dropdown_buttons(self, trace_updates: list[dict], pays_list=None, volets_list=None,
                          layout_updates: list[dict] = None) -> list[dict]:
        """
        Construit les boutons du dropdown pays/volet.

        trace_updates[i] : dict passé à args[0] du i-ème bouton (restyle sur les traces).
        layout_updates[i] : dict passé à args[1] du i-ème bouton (relayout), optionnel.
        """
        if pays_list is None:
            pays_list = PAYS_CEMAC
        if volets_list is None:
            volets_list = VOLETS
        buttons = []
        idx = 0
        for pays in pays_list:
            for volet in volets_list:
                k = (pays.lower(), volet)
                if k not in self.scores_lof and k not in self.residus and k not in self.data_brute:
                    idx += 1
                    continue
                args = [trace_updates[idx]]
                if layout_updates:
                    args.append(layout_updates[idx])
                buttons.append(dict(
                    label=self._label(pays, volet),
                    method="update",
                    args=args,
                ))
                idx += 1
        return buttons

    @staticmethod
    def _dropdown_menu(buttons: list[dict], x: float = 0.0, y: float = 1.12) -> dict:
        return dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=x,
            xanchor="left",
            y=y,
            yanchor="top",
            bgcolor="white",
            bordercolor="#ccc",
            font=dict(size=11),
        )

    # ==================================================================
    # Fig 1  Missing data map
    # ==================================================================

    def fig1_missing_data_map(self, pays: str = "cameroun", volet: str = "Actif") -> go.Figure:
        """
        Carte binaire des valeurs manquantes/invalides.
        Case noire = valeur manquante. Dropdown pays/volet intégré.

        Retourne
        --------
        go.Figure (Heatmap)
        """
        z_data, x_labels, y_labels_map = [], [], {}
        for p, v in self._all_keys():
            k = self._k(p, v)
            df = self.data_brute.get(k, pd.DataFrame())
            if df.empty:
                z_data.append(None)
            else:
                mask = df.isnull().astype(int).values.T.tolist()
                z_data.append(mask)
                if not x_labels:
                    x_labels = [_fmt_mois(d) for d in df.index]
            y_labels_map[k] = df.columns.tolist() if not df.empty else []

        k_init = self._k(pays, volet)
        df_init = self.data_brute.get(k_init, pd.DataFrame())
        z_init = df_init.isnull().astype(int).values.T.tolist() if not df_init.empty else [[]]
        x_init = [_fmt_mois(d) for d in df_init.index] if not df_init.empty else []
        y_init = df_init.columns.tolist() if not df_init.empty else []

        fig = go.Figure(go.Heatmap(
            z=z_init, x=x_init, y=y_init,
            colorscale=[[0, "white"], [1, "black"]],
            showscale=False,
            hovertemplate="Indicateur: %{y}<br>Mois: %{x}<br>Manquant: %{z}<extra></extra>",
        ))

        trace_updates, layout_updates = [], []
        for p, v in self._all_keys():
            k = self._k(p, v)
            df = self.data_brute.get(k, pd.DataFrame())
            z = df.isnull().astype(int).values.T.tolist() if not df.empty else [[]]
            x = [_fmt_mois(d) for d in df.index] if not df.empty else []
            y = df.columns.tolist() if not df.empty else []
            trace_updates.append({"z": [z], "x": [x], "y": [y]})
            layout_updates.append({"title.text": f"Fig. 1  Carte des valeurs manquantes  {self._label(p, v)}"})

        fig.update_layout(
            title=f"Fig. 1  Carte des valeurs manquantes  {self._label(pays, volet)}",
            xaxis_title="Mois",
            yaxis_title="Indicateurs",
            updatemenus=[self._dropdown_menu(
                self._dropdown_buttons(trace_updates, layout_updates=layout_updates)
            )],
            height=550,
            margin=dict(t=120, b=80, l=200),
        )
        return fig

    # ==================================================================
    # Fig 2  Heatmap des corrélations des résidus STL
    # ==================================================================

    def fig2_heatmap_correlations_stl(self, pays: str = "cameroun", volet: str = "Actif") -> go.Figure:
        """
        Heatmap de Pearson entre les résidus STL. Dropdown pays/volet.

        Retourne
        --------
        go.Figure (Heatmap)
        """
        k_init = self._k(pays, volet)
        df_init = self.residus.get(k_init, pd.DataFrame())
        corr_init = df_init.corr(method="pearson") if not df_init.empty else pd.DataFrame()

        fig = go.Figure(go.Heatmap(
            z=corr_init.values.tolist() if not corr_init.empty else [[]],
            x=corr_init.columns.tolist() if not corr_init.empty else [],
            y=corr_init.index.tolist() if not corr_init.empty else [],
            colorscale="RdBu",
            zmid=0,
            zmin=-1, zmax=1,
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="r de Pearson"),
        ))

        trace_updates, layout_updates = [], []
        for p, v in self._all_keys():
            df = self.residus.get(self._k(p, v), pd.DataFrame())
            corr = df.corr(method="pearson") if not df.empty else pd.DataFrame()
            trace_updates.append({
                "z": [corr.values.tolist() if not corr.empty else [[]]],
                "x": [corr.columns.tolist() if not corr.empty else []],
                "y": [corr.index.tolist() if not corr.empty else []],
            })
            layout_updates.append({"title.text": f"Fig. 2  Corrélations résidus STL  {self._label(p, v)}"})

        fig.update_layout(
            title=f"Fig. 2  Corrélations résidus STL  {self._label(pays, volet)}",
            updatemenus=[self._dropdown_menu(
                self._dropdown_buttons(trace_updates, layout_updates=layout_updates)
            )],
            height=650,
            margin=dict(t=120, b=120, l=150),
        )
        return fig

    # ==================================================================
    # Fig 3  Décomposition STL à 4 panneaux
    # ==================================================================

    def fig3_decomposition_stl(self, pays: str = "cameroun", volet: str = "Actif") -> go.Figure:
        """
        4 panneaux (brute / tendance / saisonnalité / résidu) pour un indicateur.
        Slider pour naviguer entre les indicateurs. Dropdown pays/volet.

        Retourne
        --------
        go.Figure (subplots 4 × 1)
        """
        COMPS = ["observed", "trend", "seasonal", "residual"]
        LABELS = ["Brute", "Tendance", "Saisonnalité", "Résidu"]
        COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        k_init = self._k(pays, volet)
        stl_init = self.data_stl.get(k_init, {})
        residus_init = self.residus.get(k_init, pd.DataFrame())
        cols_init = residus_init.columns.tolist() if not residus_init.empty else []

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=LABELS, vertical_spacing=0.04)

        # Traces initiales pour le 1er indicateur
        def _add_traces_for_key(p, v, ind_idx=0):
            k = self._k(p, v)
            stl = self.data_stl.get(k, {})
            res_df = self.residus.get(k, pd.DataFrame())
            cols = res_df.columns.tolist() if not res_df.empty else []
            ind = cols[min(ind_idx, len(cols) - 1)] if cols else None
            traces = []
            for comp, label, color in zip(COMPS, LABELS, COLORS):
                df_comp = stl.get(comp, res_df if comp == "residual" else pd.DataFrame())
                if isinstance(df_comp, pd.DataFrame) and ind and ind in df_comp.columns:
                    x = [str(d) for d in df_comp.index]
                    y = df_comp[ind].tolist()
                elif isinstance(df_comp, pd.Series):
                    x = [str(d) for d in df_comp.index]
                    y = df_comp.tolist()
                else:
                    x, y = [], []
                traces.append(go.Scatter(x=x, y=y, line=dict(color=color, width=1.5),
                                         name=label, showlegend=False))
            return traces, cols, ind

        traces_init, cols_init, ind_init = _add_traces_for_key(pays, volet, 0)
        for i, trace in enumerate(traces_init):
            fig.add_trace(trace, row=i + 1, col=1)

        # Slider pour les indicateurs
        if cols_init:
            slider_steps = []
            for ind_idx, ind_name in enumerate(cols_init):
                # Mettre à jour les 4 traces (indices 0..3)
                x_vals, y_vals = [], []
                stl = self.data_stl.get(k_init, {})
                res_df = self.residus.get(k_init, pd.DataFrame())
                for comp in COMPS:
                    df_comp = stl.get(comp, res_df if comp == "residual" else pd.DataFrame())
                    if isinstance(df_comp, pd.DataFrame) and ind_name in df_comp.columns:
                        x_vals.append([str(d) for d in df_comp.index])
                        y_vals.append(df_comp[ind_name].tolist())
                    else:
                        x_vals.append([])
                        y_vals.append([])
                slider_steps.append(dict(
                    method="restyle",
                    label=ind_name[:20],
                    args=[{"x": x_vals, "y": y_vals}, [0, 1, 2, 3]],
                ))
            fig.update_layout(
                sliders=[dict(
                    active=0,
                    steps=slider_steps,
                    currentvalue=dict(prefix="Indicateur : ", font=dict(size=11)),
                    pad=dict(t=50),
                )]
            )

        title_init = f"Fig. 3  Décomposition STL : {ind_init or ''}  {self._label(pays, volet)}"
        fig.update_layout(
            title=title_init,
            height=750,
            margin=dict(t=140, b=100),
        )
        return fig

    # ==================================================================
    # Fig 4  Scree plot RPCA
    # ==================================================================

    def fig4_scree_plot_rpca(self, pays: str = "cameroun", volet: str = "Actif") -> go.Figure:
        """
        Variance expliquée (barres) + courbe cumulée (ligne) sur axe secondaire.
        Dropdown pays/volet.

        Retourne
        --------
        go.Figure (bar + scatter dual-axis)
        """
        k_init = self._k(pays, volet)
        ve_init = self.variance_explained.get(k_init, np.array([]))

        def _series(ve):
            if ve is None or len(ve) == 0:
                return [], [], []
            k_vals = list(range(1, len(ve) + 1))
            ind_pct = (ve * 100).tolist()
            cum_pct = (np.cumsum(ve) * 100).tolist()
            return k_vals, ind_pct, cum_pct

        k_init_v, ind_init, cum_init = _series(ve_init)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=k_init_v, y=ind_init, name="Variance individuelle (%)",
                              marker_color="#4C72B0", opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=k_init_v, y=cum_init, name="Variance cumulée (%)",
                                  mode="lines+markers", line=dict(color="#DD8452", width=2)),
                      secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color="gray", secondary_y=True,
                      annotation_text="80%", annotation_position="right")

        trace_updates, layout_updates = [], []
        for p, v in self._all_keys():
            ve = self.variance_explained.get(self._k(p, v), np.array([]))
            kv, ind_p, cum_p = _series(ve)
            trace_updates.append({"x": [kv, kv], "y": [ind_p, cum_p]})
            layout_updates.append({"title.text": f"Fig. 4  Scree plot RPCA  {self._label(p, v)}"})

        fig.update_layout(
            title=f"Fig. 4  Scree plot RPCA  {self._label(pays, volet)}",
            xaxis_title="Composante",
            yaxis_title="Variance individuelle (%)",
            yaxis2_title="Variance cumulée (%)",
            legend=dict(x=0.6, y=0.5),
            updatemenus=[self._dropdown_menu(
                self._dropdown_buttons(trace_updates, layout_updates=layout_updates)
            )],
            height=480,
            margin=dict(t=120),
        )
        return fig

    # ==================================================================
    # Fig 5  Biplot ACP
    # ==================================================================

    def fig5_biplot_acp(self, pays: str = "cameroun", volet: str = "Actif") -> go.Figure:
        """
        Observations projetées (colorées par année) + vecteurs loadings.
        Slider pour la paire de composantes. Dropdown pays/volet.

        Retourne
        --------
        go.Figure (scatter)
        """
        k_init = self._k(pays, volet)
        proj_init = self.composantes_pca.get(k_init, np.array([]))
        load_init = self.loadings_pca.get(k_init, pd.DataFrame())
        df_ref_init = self.residus.get(k_init, pd.DataFrame())

        def _pairs(n_comp):
            return [(i, j) for i in range(n_comp) for j in range(i + 1, n_comp)]

        def _build_traces(proj, load_df, df_ref, i_c=0, j_c=1):
            if proj is None or proj.size == 0:
                return [go.Scatter(x=[], y=[], mode="markers")], ["PC1"], ["PC2"]
            n_obs, n_comp = proj.shape
            # Cas n_comp = 1 : affichage 1D (PC1 vs indice temporel)
            if n_comp == 1:
                years = (df_ref.index.year.tolist()
                         if not df_ref.empty and hasattr(df_ref.index, "year")
                         else list(range(n_obs)))
                x_vals = list(range(n_obs))
                y_vals = proj[:, 0].tolist()
                labels = [str(df_ref.index[k])[:7] if not df_ref.empty else str(k)
                          for k in range(n_obs)]
                return [go.Scatter(
                    x=x_vals, y=y_vals, mode="markers",
                    marker=dict(color=years, colorscale="Plasma", size=6,
                                colorbar=dict(title="Année"), showscale=True),
                    text=labels, hovertemplate="%{text}<extra></extra>",
                    name="Observations",
                )], ["Indice temporel"], ["PC1"]
            years = (df_ref.index.year.tolist()
                     if not df_ref.empty and hasattr(df_ref.index, "year")
                     else list(range(n_obs)))
            traces = [go.Scatter(
                x=proj[:, i_c].tolist(), y=proj[:, j_c].tolist(),
                mode="markers",
                marker=dict(color=years, colorscale="Plasma", size=6,
                            colorbar=dict(title="Année"), showscale=True),
                text=[str(df_ref.index[k])[:7] if not df_ref.empty else str(k) for k in range(n_obs)],
                hovertemplate="%{text}<extra></extra>",
                name="Observations",
            )]
            if not load_df.empty and i_c < len(load_df) and j_c < len(load_df):
                scale = 0.8 * max(abs(proj[:, i_c]).max(), abs(proj[:, j_c]).max())
                for col in load_df.columns:
                    lx = float(load_df.iloc[i_c][col]) * scale
                    ly = float(load_df.iloc[j_c][col]) * scale
                    traces.append(go.Scatter(
                        x=[0, lx], y=[0, ly], mode="lines+text",
                        line=dict(color="crimson", width=1),
                        text=["", col[:15]], textposition="top center",
                        textfont=dict(size=7, color="crimson"),
                        showlegend=False, hoverinfo="skip",
                    ))
            return traces, [f"PC{i_c + 1}"], [f"PC{j_c + 1}"]

        n_comp_init = proj_init.shape[1] if proj_init.size > 0 else 0
        pairs_init = _pairs(n_comp_init) if n_comp_init >= 2 else []

        # Pour n_comp = 1 : appel avec i_c=0, j_c=0 (flag → cas 1D géré dans _build_traces)
        traces_init, xlab_init, ylab_init = _build_traces(
            proj_init, load_init, df_ref_init,
            i_c=0, j_c=1 if n_comp_init >= 2 else 0
        )
        fig = go.Figure(data=traces_init)

        # Slider paires de composantes
        slider_steps = []
        for i_c, j_c in pairs_init:
            traces_step, xl, yl = _build_traces(proj_init, load_init, df_ref_init, i_c, j_c)
            xs = [t.x for t in traces_step]
            ys = [t.y for t in traces_step]
            slider_steps.append(dict(
                method="restyle",
                label=f"PC{i_c+1}–PC{j_c+1}",
                args=[{"x": xs, "y": ys}],
            ))

        fig.update_layout(
            title=f"Fig. 5  Biplot ACP  {self._label(pays, volet)}",
            xaxis_title=xlab_init[0],
            yaxis_title=ylab_init[0],
            sliders=[dict(active=0, steps=slider_steps,
                          currentvalue=dict(prefix="Paire : ", font=dict(size=11)),
                          pad=dict(t=50))] if slider_steps else [],
            height=600,
            margin=dict(t=120, b=120),
        )
        return fig

    # ==================================================================
    # Fig 6  Projection UMAP
    # ==================================================================

    def fig6_projection_umap(self, pays: str = "cameroun", volet: str = "Actif") -> go.Figure:
        """
        Embedding UMAP 2-D coloré par score LOF*. Anomalies cerclées en rouge.
        Dropdown pays/volet.

        Retourne
        --------
        go.Figure (scatter)
        """
        k_init = self._k(pays, volet)
        emb_init = self.embedding_umap.get(k_init, np.array([]))
        scores_init = self.scores_lof.get(k_init, pd.Series(dtype=float))
        tau_init = self.tau.get(k_init)
        df_ref = self.residus.get(k_init, pd.DataFrame())

        def _make_traces(emb, scores, tau_val, df_ref_local):
            if emb is None or emb.size == 0:
                return [go.Scatter(x=[], y=[], mode="markers")]
            c = scores.values if len(scores) == len(emb) else np.ones(len(emb))
            labels = ([str(df_ref_local.index[i])[:7] for i in range(len(emb))]
                      if not df_ref_local.empty else [str(i) for i in range(len(emb))])
            traces = [go.Scatter(
                x=emb[:, 0].tolist(), y=emb[:, 1].tolist(),
                mode="markers",
                marker=dict(color=c.tolist(), colorscale="Viridis", size=6,
                            colorbar=dict(title="Score LOF*"), showscale=True),
                text=labels,
                hovertemplate="%{text}<br>LOF*: %{marker.color:.3f}<extra></extra>",
                name="Observations",
            )]
            if tau_val is not None:
                mask = c > tau_val
                if mask.any():
                    traces.append(go.Scatter(
                        x=emb[mask, 0].tolist(), y=emb[mask, 1].tolist(),
                        mode="markers",
                        marker=dict(symbol="circle-open", color="red", size=14, line=dict(width=2)),
                        name=f"LOF* > τ={tau_val:.2f}",
                    ))
            return traces

        traces_init = _make_traces(emb_init, scores_init, tau_init, df_ref)
        fig = go.Figure(data=traces_init)

        trace_updates, layout_updates = [], []
        for p, v in self._all_keys():
            k = self._k(p, v)
            emb = self.embedding_umap.get(k, np.array([]))
            sc = self.scores_lof.get(k, pd.Series(dtype=float))
            tv = self.tau.get(k)
            dfr = self.residus.get(k, pd.DataFrame())
            tr = _make_traces(emb, sc, tv, dfr)
            trace_updates.append({"x": [t.x for t in tr], "y": [t.y for t in tr]})
            layout_updates.append({"title.text": f"Fig. 6  Projection UMAP  {self._label(p, v)}"})

        fig.update_layout(
            title=f"Fig. 6  Projection UMAP  {self._label(pays, volet)}",
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            updatemenus=[self._dropdown_menu(
                self._dropdown_buttons(trace_updates, layout_updates=layout_updates)
            )],
            height=550,
            margin=dict(t=120),
        )
        return fig

    # ==================================================================
    # Fig 7  Profils LOF(MinPts)  ANIMÉ
    # ==================================================================

    def fig7_profils_lof_minpts(
        self, pays: str = "cameroun", volet: str = "Actif", top_n: int = 20
    ) -> go.Figure:
        """
        Pour les top_n observations au LOF* le plus élevé : LOF(MinPts) sur [10,30].
        Animation Plotly : Play/Pause + slider déplaçant la ligne verticale MinPts.

        Retourne
        --------
        go.Figure (scatter animé)
        """
        k = self._k(pays, volet)
        df_mp = self.lof_par_minpts.get(k, pd.DataFrame())
        scores = self.scores_lof.get(k, pd.Series(dtype=float))

        if df_mp.empty or scores.empty:
            fig = go.Figure()
            fig.update_layout(title=f"Fig. 7  Aucune donnée LOF(MinPts)  {self._label(pays, volet)}")
            return fig

        top_idx = scores.dropna().nlargest(top_n).index
        df_top = df_mp.loc[df_mp.index.isin(top_idx)]
        minpts_vals = sorted(df_mp.columns.astype(int).tolist())
        colors = [f"hsl({int(i * 360 / max(top_n, 1))},70%,50%)" for i in range(len(df_top))]

        # Traces de base (lignes de profil)
        traces = []
        for i, (dt, row) in enumerate(df_top.iterrows()):
            traces.append(go.Scatter(
                x=minpts_vals,
                y=[row.get(m, None) for m in minpts_vals],
                mode="lines",
                line=dict(color=colors[i], width=1.2),
                name=_fmt_mois(dt),
            ))
        # Ligne verticale initiale (shape sur axe x)
        traces.append(go.Scatter(
            x=[minpts_vals[0], minpts_vals[0]],
            y=[0, df_top.values.max() * 1.1 if df_top.size > 0 else 1],
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name="MinPts courant",
            showlegend=True,
        ))

        # Frames pour l'animation
        n_traces_lines = len(df_top)
        frames = []
        for mp in minpts_vals:
            y_max = df_top.values.max() * 1.1 if df_top.size > 0 else 1
            frame_data = [go.Scatter(x=[mp, mp], y=[0, y_max])]
            frames.append(go.Frame(
                data=frame_data,
                traces=[n_traces_lines],
                name=str(mp),
            ))

        fig = go.Figure(data=traces, frames=frames)
        fig.update_layout(
            title=f"Fig. 7  Profils LOF(MinPts)  top {top_n}  {self._label(pays, volet)}",
            xaxis_title="MinPts",
            yaxis_title="LOF(MinPts)",
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.1, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=300, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode="immediate")]),
                ],
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[str(mp)],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                            label=str(mp)) for mp in minpts_vals],
                currentvalue=dict(prefix="MinPts = ", font=dict(size=12)),
                pad=dict(t=50),
            )],
            height=520,
            margin=dict(t=130, b=100),
        )
        return fig

    # ==================================================================
    # Fig 8  Distribution empirique des scores LOF*
    # ==================================================================

    def fig8_distribution_lof(self, pays: str = "cameroun", volet: str = "Actif") -> go.Figure:
        """
        Histogramme des scores LOF* avec seuil τ IQR et percentile 95.
        Dropdown pays/volet.

        Retourne
        --------
        go.Figure (histogram + vlines)
        """
        k_init = self._k(pays, volet)
        scores_init = self.scores_lof.get(k_init, pd.Series(dtype=float)).dropna()
        tau_init = self.tau.get(k_init)
        p95_init = float(np.percentile(scores_init.values, 95)) if not scores_init.empty else None

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scores_init.values.tolist() if not scores_init.empty else [],
            nbinsx=30,
            histnorm="probability density",
            marker_color="#4C72B0",
            opacity=0.75,
            name="LOF*",
        ))
        if tau_init is not None:
            fig.add_vline(x=tau_init, line_dash="dash", line_color="crimson", line_width=2,
                          annotation_text=f"τ IQR = {tau_init:.3f}", annotation_position="top right")
        if p95_init is not None:
            fig.add_vline(x=p95_init, line_dash="dot", line_color="darkorange", line_width=1.8,
                          annotation_text=f"p95 = {p95_init:.3f}", annotation_position="top left")

        trace_updates, layout_updates = [], []
        for p, v in self._all_keys():
            sc = self.scores_lof.get(self._k(p, v), pd.Series(dtype=float)).dropna()
            trace_updates.append({"x": [sc.values.tolist() if not sc.empty else []]})
            layout_updates.append({"title.text": f"Fig. 8  Distribution LOF*  {self._label(p, v)}"})

        fig.update_layout(
            title=f"Fig. 8  Distribution LOF*  {self._label(pays, volet)}",
            xaxis_title="Score LOF*",
            yaxis_title="Densité",
            bargap=0.05,
            updatemenus=[self._dropdown_menu(
                self._dropdown_buttons(trace_updates, layout_updates=layout_updates)
            )],
            height=450,
            margin=dict(t=120),
        )
        return fig

    # ==================================================================
    # Fig 9  Boxplot des scores LOF* par année
    # ==================================================================

    def fig9_boxplot_lof_annee(self, pays: str = "cameroun", volet: str = "Actif") -> go.Figure:
        """
        Calendrier heatmap LOF* : années (lignes) × mois (colonnes).
        Cercles rouges = mois anomaux (LOF* > τ). Dropdown pays/volet.

        Retourne
        --------
        go.Figure (Heatmap + Scatter overlay)
        """
        k_init = self._k(pays, volet)
        scores_init = self.scores_lof.get(k_init, pd.Series(dtype=float)).dropna()
        tau_init = self.tau.get(k_init)

        def _cal_matrix(sc, tau_v):
            """Build (z_matrix, years, anom_x_labels, anom_y_labels) for calendar heatmap."""
            if sc.empty or not hasattr(sc.index, "year"):
                return None, None, [], []
            years = sorted(sc.index.year.unique())
            z = np.full((len(years), 12), np.nan)
            for i, yr in enumerate(years):
                yr_sc = sc[sc.index.year == yr]
                for dt, val in yr_sc.items():
                    z[i, dt.month - 1] = val
            anom_x, anom_y = [], []
            if tau_v is not None:
                for i, yr in enumerate(years):
                    for m in range(12):
                        if not np.isnan(z[i, m]) and z[i, m] > tau_v:
                            anom_x.append(_MOIS_LABELS[m])
                            anom_y.append(str(yr))
            return z, [str(yr) for yr in years], anom_x, anom_y

        z_init, y_init, ax_init, ay_init = _cal_matrix(scores_init, tau_init)

        if z_init is None:
            fig = go.Figure()
            fig.update_layout(title=f"Fig. 9  Calendrier LOF*  {self._label(pays, volet)}")
            return fig

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=z_init,
            x=_MOIS_LABELS,
            y=y_init,
            colorscale="YlOrRd",
            colorbar=dict(title="Score LOF*"),
            hovertemplate="Année %{y}  %{x} : LOF* = %{z:.3f}<extra></extra>",
            name="LOF*",
        ))
        fig.add_trace(go.Scatter(
            x=ax_init, y=ay_init,
            mode="markers",
            marker=dict(symbol="circle-open", size=16, color="crimson",
                        line=dict(width=2.5)),
            name="Anomalie",
            hovertemplate="Anomalie  %{y}  %{x}<extra></extra>",
        ))

        # Build dropdown manually (2 traces per combo)
        all_combo_keys = [(p, v) for p in PAYS_CEMAC for v in VOLETS]
        avail = [(p, v) for p, v in all_combo_keys if self._k(p, v) in self.scores_lof]
        n_total = len(avail) * 2  # heatmap + scatter per combo (pre-built traces)

        # Replace with a simpler approach: store all traces, toggle visibility
        fig.data[0].visible = True
        fig.data[1].visible = True

        buttons = []
        init_k = k_init
        for p, v in avail:
            k = self._k(p, v)
            sc = self.scores_lof.get(k, pd.Series(dtype=float)).dropna()
            tv = self.tau.get(k)
            z, y_lbl, ax, ay = _cal_matrix(sc, tv)
            if z is None:
                z, y_lbl, ax, ay = np.array([[]]), [], [], []
            tau_str = f"  τ = {tv:.3f}" if tv is not None else ""
            buttons.append(dict(
                label=self._label(p, v),
                method="update",
                args=[
                    {
                        "z": [z.tolist(), None],
                        "x": [_MOIS_LABELS, ax],
                        "y": [y_lbl, ay],
                    },
                    {"title.text": f"Fig. 9  Calendrier LOF* par mois/année  "
                                   f"{self._label(p, v)}{tau_str}"},
                ],
            ))

        tau_str_init = f"  τ = {tau_init:.3f}" if tau_init is not None else ""
        fig.update_layout(
            title=f"Fig. 9  Calendrier LOF* par mois/année  "
                  f"{self._label(pays, volet)}{tau_str_init}",
            xaxis_title="Mois",
            yaxis_title="Année",
            updatemenus=[self._dropdown_menu(buttons)],
            height=480,
            margin=dict(t=120, b=60),
        )
        return fig

    # ==================================================================
    # Fig 10  Série temporelle LOF*(t)  ANIMÉ
    # ==================================================================

    def fig10_serie_temporelle_lof(
        self, pays: str = "cameroun", volet: str = "Actif", animate: bool = True
    ) -> go.Figure:
        """
        LOF*(t) sur l'axe du temps, ligne τ, points anomaux annotés.
        Si animate=True : révélation progressive de la série.

        Retourne
        --------
        go.Figure (scatter animé ou statique)
        """
        k = self._k(pays, volet)
        scores = self.scores_lof.get(k, pd.Series(dtype=float)).dropna()
        tau_val = self.tau.get(k)

        if scores.empty:
            fig = go.Figure()
            fig.update_layout(title=f"Fig. 10  Aucun score LOF*  {self._label(pays, volet)}")
            return fig

        dates = [_fmt_mois(d) for d in scores.index]
        vals = scores.values.tolist()

        if not animate:
            # Version statique avec dropdown tous pays/volets
            all_combo_keys = [(p, v) for p in PAYS_CEMAC for v in VOLETS]

            # Combos disponibles dans l'ordre PAYS_CEMAC × VOLETS
            avail = [(ci, p, v) for ci, (p, v) in enumerate(all_combo_keys)
                     if self._k(p, v) in self.scores_lof]
            n_per    = 3          # traces par combo : ligne LOF*, marqueurs anomalies, ligne τ
            n_traces = len(avail) * n_per

            traces  = []
            buttons = []
            init_k  = self._k(pays, volet)

            for seq, (ci, p, v) in enumerate(avail):
                k_i    = self._k(p, v)
                sc_i   = self.scores_lof[k_i].dropna()
                tau_i  = self.tau.get(k_i)
                is_ini = (k_i == init_k)

                dates_i = [_fmt_mois(d) for d in sc_i.index]
                vals_i  = sc_i.values.tolist()

                # Trace 0  courbe LOF*
                traces.append(go.Scatter(
                    x=dates_i, y=vals_i, mode="lines",
                    line=dict(color="#1f77b4", width=1.5),
                    name="LOF*", visible=is_ini, showlegend=False,
                ))

                # Trace 1  points anomaux
                if tau_i is not None and vals_i:
                    msk    = np.array(vals_i) > tau_i
                    anom_x = [dates_i[j] for j in range(len(dates_i)) if msk[j]]
                    anom_y = [vals_i[j]  for j in range(len(vals_i))  if msk[j]]
                else:
                    anom_x, anom_y = [], []
                traces.append(go.Scatter(
                    x=anom_x, y=anom_y, mode="markers",
                    marker=dict(color="crimson", size=8, symbol="circle"),
                    name="Anomalie", visible=is_ini, showlegend=False,
                ))

                # Trace 2  ligne τ (scatter, pour être contrôlable par dropdown)
                if tau_i is not None and dates_i:
                    tau_x = [dates_i[0], dates_i[-1]]
                    tau_y = [tau_i, tau_i]
                    tau_t = ["", f"τ = {tau_i:.3f}"]
                else:
                    tau_x, tau_y, tau_t = [], [], []
                traces.append(go.Scatter(
                    x=tau_x, y=tau_y, mode="lines+text",
                    line=dict(color="crimson", dash="dash", width=1.5),
                    text=tau_t, textposition="middle right",
                    name=f"τ = {tau_i:.3f}" if tau_i else "τ",
                    visible=is_ini, showlegend=False,
                ))

                # Bouton dropdown  rend visibles uniquement les 3 traces de ce combo
                vis = [False] * n_traces
                vis[seq * n_per]     = True
                vis[seq * n_per + 1] = True
                vis[seq * n_per + 2] = True
                tau_str = f"  τ = {tau_i:.3f}" if tau_i else ""
                buttons.append(dict(
                    label=self._label(p, v),
                    method="update",
                    args=[
                        {"visible": vis},
                        {"title.text": f"Fig. 10  Série temporelle LOF*  "
                                       f"{self._label(p, v)}{tau_str}"},
                    ],
                ))

            tau_str_init = f"  τ = {self.tau[init_k]:.3f}" if init_k in self.tau else ""
            fig = go.Figure(data=traces)
            fig.update_layout(
                title=f"Fig. 10  Série temporelle LOF*  "
                      f"{self._label(pays, volet)}{tau_str_init}",
                xaxis_title="Date", yaxis_title="Score LOF*", height=450,
                margin=dict(t=110),
                updatemenus=[self._dropdown_menu(buttons)],
            )
            return fig

        # Version animée : frames de 1 à N observations
        step = max(1, len(dates) // 80)
        frames = []
        for end in range(step, len(dates) + 1, step):
            frame_data = [go.Scatter(x=dates[:end], y=vals[:end])]
            frames.append(go.Frame(data=frame_data, name=str(end)))

        fig = go.Figure(
            data=[go.Scatter(x=dates[:step], y=vals[:step],
                              mode="lines", line=dict(color="#1f77b4", width=1.5), name="LOF*")],
            frames=frames,
        )
        if tau_val is not None:
            fig.add_hline(y=tau_val, line_dash="dash", line_color="crimson",
                          annotation_text=f"τ = {tau_val:.3f}")

        slider_steps = []
        for i, end in enumerate(range(step, len(dates) + 1, step), start=1):
            lbl = dates[end - 1] if end - 1 < len(dates) else dates[-1]
            slider_steps.append(dict(
                method="animate", label=lbl,
                args=[[str(end)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
            ))

        fig.update_layout(
            title=f"Fig. 10  Série temporelle LOF* (animée)  {self._label(pays, volet)}",
            xaxis_title="Mois", yaxis_title="Score LOF*",
            yaxis=dict(range=[0, max(vals) * 1.15]),
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.1, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(mode="immediate", frame=dict(duration=0))]),
                ],
            )],
            sliders=[dict(
                steps=slider_steps,
                currentvalue=dict(prefix="Mois : "),
                pad=dict(t=50),
            )],
            height=480, margin=dict(t=130, b=100),
        )
        return fig

    # ==================================================================
    # Fig 11  Superposition LOF* + indicateur brut
    # ==================================================================

    def fig11_superposition_lof_indicateur(
        self, pays: str = "cameroun", volet: str = "Actif"
    ) -> go.Figure:
        """
        Double axe : LOF*(t) à gauche, indicateur brut à droite.
        Slider pour l'indicateur. Dropdown pays/volet.

        Retourne
        --------
        go.Figure (dual-axis scatter)
        """
        k_init = self._k(pays, volet)
        scores_init = self.scores_lof.get(k_init, pd.Series(dtype=float)).dropna()
        df_brut_init = self.data_brute.get(k_init, pd.DataFrame())
        cols_init = df_brut_init.columns.tolist() if not df_brut_init.empty else []
        tau_init = self.tau.get(k_init)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        def _add_traces(sc, df_b, tau_v, ind_idx=0):
            traces = []
            x_dates = [_fmt_mois(d) for d in sc.index] if not sc.empty else []
            traces.append(go.Scatter(
                x=x_dates, y=sc.values.tolist() if not sc.empty else [],
                name="LOF*", line=dict(color="#1f77b4", width=1.5), yaxis="y",
            ))
            cols = df_b.columns.tolist() if not df_b.empty else []
            if cols:
                ind = cols[min(ind_idx, len(cols) - 1)]
                serie = df_b[ind].reindex(sc.index) if not sc.empty else df_b[ind]
                traces.append(go.Scatter(
                    x=x_dates, y=serie.values.tolist(),
                    name=ind, line=dict(color="#ff7f0e", width=1.2, dash="dash"), yaxis="y2",
                ))
            return traces

        for trace in _add_traces(scores_init, df_brut_init, tau_init, 0):
            fig.add_trace(trace, secondary_y=(trace.yaxis == "y2" if hasattr(trace, "yaxis") else False))

        # Slider indicateurs
        if cols_init:
            steps = []
            for ind_idx, ind_name in enumerate(cols_init):
                sc = scores_init
                serie = df_brut_init[ind_name].reindex(sc.index)
                steps.append(dict(
                    method="restyle",
                    label=ind_name[:20],
                    args=[{"y": [sc.values.tolist(), serie.values.tolist()]}, [0, 1]],
                ))
            fig.update_layout(sliders=[dict(
                active=0, steps=steps,
                currentvalue=dict(prefix="Indicateur : "),
                pad=dict(t=50),
            )])

        if tau_init is not None:
            fig.add_hline(y=tau_init, line_dash="dash", line_color="#1f77b4",
                          opacity=0.5, annotation_text=f"τ = {tau_init:.3f}")

        fig.update_layout(
            title=f"Fig. 11  LOF* vs Indicateur brut  {self._label(pays, volet)}",
            xaxis_title="Date",
            yaxis_title="Score LOF*",
            yaxis2_title="Valeur indicateur",
            height=480,
            margin=dict(t=120, b=100),
        )
        return fig

    # ==================================================================
    # Fig 12  Radar chart des indicateurs contributeurs
    # ==================================================================

    def fig12_bar_chart_indicateurs(
        self, pays: str = "cameroun", volet: str = "Actif", top_n: int = 15
    ) -> go.Figure:
        """
        Bar chart horizontal  top-15 indicateurs contributeurs pour les mois anomaux.

        Spécification graphique.tex §Fig. 12 :
        - Barres triées par |contribution| décroissante (top 15)
        - Rouge (#E05252) : contribution positive à l'anomalie (résidu > 0)
        - Bleu  (#5BA8E5) : contribution négative (résidu < 0)
        - Slider pour naviguer entre les mois anomaux
        - Dropdown pays/volet

        Retourne
        --------
        go.Figure (bar horizontal)
        """
        k_init = self._k(pays, volet)
        scores_init = self.scores_lof.get(k_init, pd.Series(dtype=float)).dropna()
        res_init    = self.residus.get(k_init, pd.DataFrame())
        tau_init    = self.tau.get(k_init)

        def _anom_months(sc, res_df, tau_v):
            if sc.empty or res_df.empty or tau_v is None:
                return []
            return [dt for dt in sc[sc > tau_v].index if dt in res_df.index]

        def _bar_traces(res_df, dt, n):
            """Return (y_labels, x_values, colors) for the top-n indicators."""
            row = res_df.loc[dt]
            top = row.abs().nlargest(n)
            indicators = top.index.tolist()[::-1]          # ascending for horizontal bar
            values     = [row[ind] for ind in indicators]
            colors     = ["#E05252" if v > 0 else "#5BA8E5" for v in values]
            return indicators, values, colors

        anom_init = _anom_months(scores_init, res_init, tau_init)

        if anom_init:
            y0, x0, c0 = _bar_traces(res_init, anom_init[0], top_n)
            score0 = scores_init.get(anom_init[0], float("nan"))
            subtitle = f"Mois : {_fmt_mois(anom_init[0])}  |  LOF* = {score0:.2f}"
        else:
            y0, x0, c0 = [], [], []
            subtitle = "Aucune anomalie détectée"

        fig = go.Figure(go.Bar(
            x=x0, y=y0,
            orientation="h",
            marker_color=c0,
            text=[f"{v:+.3f}" for v in x0],
            textposition="outside",
            name="",
        ))

        # Slider  one step per anomalous month
        if anom_init:
            steps = []
            for dt in anom_init:
                y, x, c = _bar_traces(res_init, dt, top_n)
                sc = scores_init.get(dt, float("nan"))
                steps.append(dict(
                    method="update",
                    label=_fmt_mois(dt),
                    args=[
                        {"x": [x], "y": [y], "marker.color": [c],
                         "text": [[f"{v:+.3f}" for v in x]]},
                        {"title": {
                            "text": (
                                f"Fig. 12  Indicateurs contributeurs  "
                                f"{self._label(pays, volet)}<br>"
                                f"<sup>Mois : {_fmt_mois(dt)}  |  LOF* = {sc:.2f}</sup>"
                            )
                        }},
                    ],
                ))
            fig.update_layout(sliders=[dict(
                active=0, steps=steps,
                currentvalue=dict(prefix="Mois : ", font=dict(size=12)),
                pad=dict(t=60, b=10),
            )])

        fig.update_layout(
            title=dict(
                text=(
                    f"Fig. 12  Indicateurs contributeurs  "
                    f"{self._label(pays, volet)}<br>"
                    f"<sup>{subtitle}</sup>"
                ),
                font=dict(size=14),
            ),
            xaxis=dict(
                title="Résidu normalisé (contribution à l'anomalie)",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.2)",
            ),
            yaxis=dict(title="Indicateur IFS", automargin=True),
            height=560,
            margin=dict(t=120, b=80, l=160, r=80),
        )
        return fig

    # ==================================================================
    # Fig 13  Heatmap des anomalies (indicateurs × mois anomaux)
    # ==================================================================

    def fig13_heatmap_anomalies(self, pays: str = "cameroun", volet: str = "Actif") -> go.Figure:
        """
        Heatmap déviation résiduelle pour les mois anomaux uniquement.
        Dropdown pays/volet.

        Retourne
        --------
        go.Figure (Heatmap)
        """
        k_init = self._k(pays, volet)
        scores_init = self.scores_lof.get(k_init, pd.Series(dtype=float)).dropna()
        res_init = self.residus.get(k_init, pd.DataFrame())
        tau_init = self.tau.get(k_init)

        def _sub(sc, res_df, tv):
            if sc.empty or res_df.empty or tv is None:
                return pd.DataFrame()
            anom = sc[sc > tv].index
            return res_df.loc[res_df.index.isin(anom)]

        def _cluster_indicators(sub: pd.DataFrame) -> pd.DataFrame:
            """Sort indicators (columns) by Ward hierarchical clustering on anomaly profiles."""
            if sub.empty or sub.shape[1] < 2:
                return sub
            if sub.shape[1] > 50:
                return sub  # skip clustering pour éviter timeout O(n²log n)
            try:
                from scipy.cluster.hierarchy import linkage, leaves_list
                from scipy.spatial.distance import pdist
                mat = sub.T.fillna(0).values   # (n_indicators, n_anomaly_months)
                dist = pdist(mat, metric="euclidean")
                if len(dist) == 0 or np.any(np.isnan(dist)):
                    return sub
                Z = linkage(dist, method="ward")
                order = leaves_list(Z)
                return sub.iloc[:, order]
            except ImportError:
                return sub   # scipy absent → keep original order

        sub_init = _cluster_indicators(_sub(scores_init, res_init, tau_init))
        vmax_init = float(max(abs(sub_init.values.max()), abs(sub_init.values.min()))) if not sub_init.empty else 1

        fig = go.Figure(go.Heatmap(
            z=sub_init.T.values.tolist() if not sub_init.empty else [[]],
            x=[_fmt_mois(d) for d in sub_init.index] if not sub_init.empty else [],
            y=sub_init.columns.tolist() if not sub_init.empty else [],
            colorscale="RdBu",
            zmid=0, zmin=-vmax_init, zmax=vmax_init,
            colorbar=dict(title="Déviation résiduelle"),
            hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
        ))

        trace_updates, layout_updates = [], []
        for p, v in self._all_keys():
            k = self._k(p, v)
            sc = self.scores_lof.get(k, pd.Series(dtype=float)).dropna()
            res = self.residus.get(k, pd.DataFrame())
            tv = self.tau.get(k)
            sub = _cluster_indicators(_sub(sc, res, tv))
            vm = float(max(abs(sub.values.max()), abs(sub.values.min()))) if not sub.empty else 1
            trace_updates.append({
                "z": [sub.T.values.tolist() if not sub.empty else [[]]],
                "x": [[_fmt_mois(d) for d in sub.index] if not sub.empty else []],
                "y": [sub.columns.tolist() if not sub.empty else []],
                "zmin": [-vm], "zmax": [vm],
            })
            layout_updates.append({"title.text": f"Fig. 13  Heatmap anomalies  {self._label(p, v)}"})

        fig.update_layout(
            title=f"Fig. 13  Heatmap anomalies  {self._label(pays, volet)}",
            updatemenus=[self._dropdown_menu(
                self._dropdown_buttons(trace_updates, layout_updates=layout_updates)
            )],
            height=600,
            margin=dict(t=120, b=100, l=200),
        )
        return fig

    # ==================================================================
    # Fig 14  Carte de chaleur inter-pays  ANIMÉE
    # ==================================================================

    def fig14_heatmap_interpays(self, volet: str = "Actif", animate: bool = True) -> go.Figure:
        """
        Heatmap mois × pays CEMAC, couleur = LOF* normalisé par pays.
        Animation : révélation colonne par colonne (mois à mois).
        Dropdown pour choisir le volet (Actif / Passif).

        Retourne
        --------
        go.Figure (heatmap animé ou statique)
        """
        def _build_matrix(v):
            series_list = []
            for p in PAYS_CEMAC:
                sc = self.scores_lof.get(self._k(p, v), pd.Series(dtype=float))
                if not sc.empty:
                    sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
                    sc_norm.name = p
                    series_list.append(sc_norm)
            if not series_list:
                return pd.DataFrame()
            return pd.concat(series_list, axis=1).sort_index()

        mat = _build_matrix(volet)

        if mat.empty:
            fig = go.Figure()
            fig.update_layout(title=f"Fig. 14  Aucune donnée  volet {volet}")
            return fig

        x_dates = [_fmt_mois(d) for d in mat.index]
        y_pays = mat.columns.tolist()
        z_full = mat.T.values.tolist()

        if not animate:
            # Count countries above their national threshold per month (aux panel)
            count_per_month = pd.Series(0, index=mat.index, dtype=int)
            for p in PAYS_CEMAC:
                k_p = self._k(p, volet)
                sc_p = self.scores_lof.get(k_p)
                tau_p = self.tau.get(k_p)
                if sc_p is not None and tau_p is not None:
                    above = (sc_p > tau_p).reindex(mat.index).fillna(False)
                    count_per_month += above.astype(int)

            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.83, 0.17],
                horizontal_spacing=0.03,
                subplot_titles=["LOF* normalisé (min-max par pays)", "Pays en alerte"],
            )

            # Left: heatmap
            fig.add_trace(go.Heatmap(
                z=z_full, x=x_dates, y=y_pays,
                colorscale="YlOrRd", zmin=0, zmax=1,
                colorbar=dict(title="LOF* normalisé", len=0.8, x=0.80),
                hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
                showscale=True,
            ), row=1, col=1)

            # Right: count bar chart (grey) with systemic threshold line at m=3
            fig.add_trace(go.Bar(
                y=x_dates,
                x=count_per_month.values,
                orientation="h",
                marker_color="rgba(200,200,200,0.5)",
                name="Pays en alerte",
                showlegend=False,
                hovertemplate="%{y}: %{x} pays<extra></extra>",
            ), row=1, col=2)
            fig.add_vline(
                x=3, line_dash="dash", line_color="#E05252", line_width=1.5,
                annotation_text="Seuil systémique (m≥3)",
                annotation_font_size=8, annotation_position="top right",
                row=1, col=2,
            )
            fig.update_xaxes(range=[0, len(PAYS_CEMAC)], row=1, col=2)
            fig.update_layout(
                title=f"Fig. 14  Carte inter-pays LOF*  {volet}",
                height=500, margin=dict(t=100, b=100),
                bargap=0.1,
            )
            return fig

        # Version animée : colonnes révélées progressivement
        z_mat = mat.T.values  # shape (n_pays, n_mois)
        n_pays, n_mois = z_mat.shape
        step = max(1, n_mois // 60)
        frames = []
        for end in range(step, n_mois + 1, step):
            z_frame = np.full_like(z_mat, np.nan)
            z_frame[:, :end] = z_mat[:, :end]
            frames.append(go.Frame(
                data=[go.Heatmap(z=z_frame.tolist(), x=x_dates, y=y_pays)],
                name=str(end),
            ))

        z_init = np.full_like(z_mat, np.nan)
        z_init[:, :step] = z_mat[:, :step]

        fig = go.Figure(
            data=[go.Heatmap(
                z=z_init.tolist(), x=x_dates, y=y_pays,
                colorscale="YlOrRd", zmin=0, zmax=1,
                colorbar=dict(title="LOF* normalisé"),
            )],
            frames=frames,
        )

        # Dropdown Actif / Passif
        mat_passif = _build_matrix("Passif")
        buttons_volet = []
        for v_btn, m_btn in [("Actif", mat), ("Passif", mat_passif)]:
            if not m_btn.empty:
                buttons_volet.append(dict(
                    label=v_btn,
                    method="update",
                    args=[{"z": [m_btn.T.values.tolist()],
                           "x": [[_fmt_mois(d) for d in m_btn.index]],
                           "y": [m_btn.columns.tolist()]},
                          {"title.text": f"Fig. 14  Carte inter-pays LOF*  {v_btn}"}],
                ))

        fig.update_layout(
            title=f"Fig. 14  Carte inter-pays LOF* (animée)  {volet}",
            updatemenus=[
                self._dropdown_menu(buttons_volet, x=0.85, y=1.12),
                dict(
                    type="buttons", showactive=False, y=1.1, x=0.5, xanchor="center",
                    buttons=[
                        dict(label="▶ Play", method="animate",
                             args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], dict(mode="immediate", frame=dict(duration=0))]),
                    ],
                ),
            ],
            sliders=[dict(
                steps=[dict(method="animate", label=x_dates[min(i * step, n_mois - 1)],
                            args=[[str(i * step + step)],
                                  dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                       for i in range(len(frames))],
                currentvalue=dict(prefix="Jusqu'à : "),
                pad=dict(t=50),
            )],
            height=520,
            margin=dict(t=140, b=100),
        )
        return fig

    # ==================================================================
    # Fig A  Variance RPCA sur fenêtres glissantes (36 mois)
    # ==================================================================

    def figA_variance_rpca_temporelle(
        self, pays: str = "cameroun", volet: str = "Actif",
        window: int = 36, n_comps: int = 5
    ) -> go.Figure:
        """
        Variance cumulée des n_comps premières composantes RPCA calculée sur des
        fenêtres glissantes de `window` mois (pas = 1 mois).

        Spécification graphique.tex §Fig. A :
        - Courbes une par composante (cumulatives)
        - Ligne horizontale à 90% (seuil de rétention ACP)
        - Annotations verticales des ruptures documentées :
            · Choc pétrolier 2015-01
            · Crise camerounaise 2017-10
            · COVID-19 2020-03
        - Détecte la non-stationnarité de la covariance dans le temps

        Retourne
        --------
        go.Figure (scatter)
        """
        k = self._k(pays, volet)
        res_df = self.residus.get(k, pd.DataFrame())
        if res_df.empty or len(res_df) < window + n_comps:
            fig = go.Figure()
            fig.update_layout(title="Fig. A  Données insuffisantes", height=400)
            return fig

        import torch

        dates_center = res_df.index[window - 1:]
        X_full = torch.tensor(res_df.values, dtype=torch.float32)

        try:
            # Toutes les fenêtres glissantes empilées en un seul tenseur batché
            # (n_windows, window, p) → une seule SVD batchée au lieu d'une SVD
            # Python par fenêtre (~150-300 appels), même résultat numérique.
            blocks = X_full.unfold(0, window, 1).permute(0, 2, 1).contiguous()
            blocks = blocks - blocks.mean(dim=1, keepdim=True)
            _, S, _ = torch.linalg.svd(blocks, full_matrices=False)   # S : (n_windows, k)
            k = S.shape[1]
            total = (S ** 2).sum(dim=1)                                # (n_windows,)
            var_cum = (S ** 2).cumsum(dim=1)                           # (n_windows, k)
            n_take  = min(n_comps, k)
            explained = var_cum[:, :n_take] / total.clamp(min=1e-12).unsqueeze(1)
            if n_take < n_comps:
                pad = explained[:, -1:].expand(-1, n_comps - n_take)
                explained = torch.cat([explained, pad], dim=1)
            explained = explained.clone()
            explained[total == 0] = 0.0
            cum_var_per_window = explained.tolist()
        except Exception:
            logging.getLogger(__name__).warning(
                "figA : SVD batchée échouée, repli sur boucle par fenêtre")
            cum_var_per_window = []
            for start in range(len(res_df) - window + 1):
                block = X_full[start: start + window]
                block = block - block.mean(dim=0)
                try:
                    _, S, _ = torch.linalg.svd(block, full_matrices=False)
                except Exception:
                    cum_var_per_window.append([float("nan")] * n_comps)
                    continue
                tot = (S ** 2).sum().item()
                if tot == 0:
                    cum_var_per_window.append([0.0] * n_comps)
                    continue
                expl = [(S[:i + 1] ** 2).sum().item() / tot for i in range(min(n_comps, len(S)))]
                while len(expl) < n_comps:
                    expl.append(expl[-1] if expl else 0.0)
                cum_var_per_window.append(expl)

        cum_var = list(zip(*cum_var_per_window))  # shape: (n_comps, n_windows)

        fig = go.Figure()
        palette = ["#D4A020", "#5BA8E5", "#52C97E", "#E05252", "#9b59b6"]
        for i in range(n_comps):
            fig.add_trace(go.Scatter(
                x=dates_center,
                y=list(cum_var[i]),
                name=f"CP 1–{i + 1}",
                line=dict(color=palette[i % len(palette)], width=1.5),
                mode="lines",
            ))

        # 90% threshold line
        fig.add_hline(y=0.9, line_dash="dash", line_color="rgba(255,255,255,0.35)",
                      annotation_text="90 %", annotation_position="left")

        # Structural break annotations
        ruptures = [
            ("2015-01-01", "Choc pétrolier"),
            ("2017-10-01", "Crise CMR"),
            ("2020-03-01", "COVID-19"),
        ]
        for date_str, label in ruptures:
            try:
                dt = pd.Timestamp(date_str)
                if dates_center[0] <= dt <= dates_center[-1]:
                    fig.add_vline(
                        x=dt, line_dash="dot",
                        line_color="rgba(224,82,82,0.55)",
                        annotation_text=label,
                        annotation_position="top right",
                        annotation_font_size=9,
                    )
            except Exception:
                pass

        fig.update_layout(
            title=dict(
                text=(
                    f"Fig. A  Variance expliquée RPCA  fenêtres glissantes {window} mois<br>"
                    f"<sup>{self._label(pays, volet)}</sup>"
                ),
                font=dict(size=14),
            ),
            xaxis=dict(title="Date (centre de fenêtre)"),
            yaxis=dict(title="Variance cumulée expliquée", tickformat=".0%", range=[0, 1.05]),
            height=480,
            margin=dict(t=100, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    # ==================================================================
    # Fig B  Concordance Actif–Passif par pays
    # ==================================================================

    def figB_concordance_actif_passif(self, pays: str = "cameroun") -> go.Figure:
        """
        Deux panneaux empilés (LOF* Actif / LOF* Passif) sur le même axe temporel,
        même seuil τ.

        Spécification graphique.tex §Fig. B :
        - Zones rouges  : anomalie concordante  (Actif > τ ET Passif > τ) → événement réel
        - Zones oranges : anomalie discordante  (un seul volet > τ) → vérification requise
        - Ligne τ commune (moyenne des deux seuils) en pointillés
        - Subplot 2×1 partageant l'axe x

        Retourne
        --------
        go.Figure (2 subplots)
        """
        from plotly.subplots import make_subplots

        k_a = self._k(pays, "Actif")
        k_p = self._k(pays, "Passif")
        sc_a = self.scores_lof.get(k_a, pd.Series(dtype=float)).dropna()
        sc_p = self.scores_lof.get(k_p, pd.Series(dtype=float)).dropna()
        tau_a = self.tau.get(k_a)
        tau_p = self.tau.get(k_p)

        label = self._label(pays, "Actif/Passif")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.5],
            vertical_spacing=0.06,
            subplot_titles=["Volet Actif", "Volet Passif"],
        )

        gold = "#D4A020"
        red  = "#E05252"
        orange = "#E07B30"

        def _add_panel(row, sc, tau_v, label_v):
            if sc.empty:
                return
            fig.add_trace(go.Scatter(
                x=sc.index, y=sc.values,
                name=f"LOF* {label_v}",
                line=dict(color=gold, width=1.4),
                mode="lines",
            ), row=row, col=1)

            if tau_v is not None:
                fig.add_hline(y=tau_v, line_dash="dash",
                              line_color="rgba(255,255,255,0.35)",
                              annotation_text=f"τ = {tau_v:.2f}",
                              annotation_position="top right",
                              row=row, col=1)

        _add_panel(1, sc_a, tau_a, "Actif")
        _add_panel(2, sc_p, tau_p, "Passif")

        # Shade concordance / discordance zones
        if not sc_a.empty and not sc_p.empty and tau_a is not None and tau_p is not None:
            common_idx = sc_a.index.intersection(sc_p.index)
            sc_a_c = sc_a.reindex(common_idx)
            sc_p_c = sc_p.reindex(common_idx)
            anom_a = sc_a_c > tau_a
            anom_p = sc_p_c > tau_p
            concordant   = anom_a & anom_p
            discordant_a = anom_a & ~anom_p
            discordant_p = ~anom_a & anom_p

            def _shade(mask, color, name_str):
                if not mask.any():
                    return
                for start_i, row_i in [(1, "Actif"), (2, "Passif")]:
                    in_zone = False
                    z_start = None
                    for dt, flag in mask.items():
                        if flag and not in_zone:
                            in_zone = True
                            z_start = dt
                        elif not flag and in_zone:
                            in_zone = False
                            fig.add_vrect(x0=z_start, x1=dt,
                                          fillcolor=color, opacity=0.18,
                                          line_width=0, row=start_i, col=1)
                    if in_zone and z_start is not None:
                        fig.add_vrect(x0=z_start, x1=mask.index[-1],
                                      fillcolor=color, opacity=0.18,
                                      line_width=0, row=start_i, col=1)

            _shade(concordant,   red,    "Concordante")
            _shade(discordant_a, orange, "Discordante-A")
            _shade(discordant_p, orange, "Discordante-P")

        fig.update_layout(
            title=dict(
                text=(
                    f"Fig. B  Concordance Actif–Passif  {label}<br>"
                    f"<sup>Rouge : anomalie concordante · Orange : anomalie discordante</sup>"
                ),
                font=dict(size=14),
            ),
            height=520,
            margin=dict(t=110, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="LOF*")
        return fig

    # ==================================================================
    # Export  toutes les figures
    # ==================================================================

    def generer_toutes(
        self, pays: str = "cameroun", volet: str = "Actif",
        dossier_export: str = "figures", format: str = "html"
    ) -> dict:
        """
        Génère et sauvegarde les 14 figures dans dossier_export.
        format : "html" (interactif, défaut) ou "png" (kaleido requis).

        Retourne
        --------
        dict[str -> go.Figure]
        """
        import os
        os.makedirs(dossier_export, exist_ok=True)

        methodes = {
            "fig1":  lambda: self.fig1_missing_data_map(pays, volet),
            "fig2":  lambda: self.fig2_heatmap_correlations_stl(pays, volet),
            "fig3":  lambda: self.fig3_decomposition_stl(pays, volet),
            "fig4":  lambda: self.fig4_scree_plot_rpca(pays, volet),
            "fig5":  lambda: self.fig5_biplot_acp(pays, volet),
            "fig6":  lambda: self.fig6_projection_umap(pays, volet),
            "fig7":  lambda: self.fig7_profils_lof_minpts(pays, volet),
            "fig8":  lambda: self.fig8_distribution_lof(pays, volet),
            "fig9":  lambda: self.fig9_boxplot_lof_annee(pays, volet),
            "fig10": lambda: self.fig10_serie_temporelle_lof(pays, volet, animate=False),
            "fig11": lambda: self.fig11_superposition_lof_indicateur(pays, volet),
            "fig12": lambda: self.fig12_bar_chart_indicateurs(pays, volet),
            "fig13": lambda: self.fig13_heatmap_anomalies(pays, volet),
            "fig14": lambda: self.fig14_heatmap_interpays(volet, animate=False),
            "figA":  lambda: self.figA_variance_rpca_temporelle(pays, volet),
            "figB":  lambda: self.figB_concordance_actif_passif(pays),
        }
        resultats = {}
        for nom, fn in methodes.items():
            fig = fn()
            if format == "html":
                path = os.path.join(dossier_export, f"{nom}_{pays}_{volet}.html")
                fig.write_html(path, include_plotlyjs="cdn")
            else:
                path = os.path.join(dossier_export, f"{nom}_{pays}_{volet}.png")
                fig.write_image(path, scale=2)
            resultats[nom] = fig
        return resultats
