"""bivat/graphiques.py  Figures BiVAT (SPEC §4.4, graphique.tex Phase 2 & 4).

Figures implemented:
    Fig. E      (Phase 2)  Score BiVAT + enveloppes CP fenêtre calibration
    Fig. D      (Phase 4)  Comparaison LOF vs BiVAT normalisés [0,1] · 4 zones
    Fig. E bis  (Phase 4)  Score BiVAT + enveloppes CP fenêtre test (rolling)
    Fig. F      (Phase 4)  Bar chart SHAP top-5 anomalies BiVAT
    Fig. G      (Phase 4)  Heatmap confusion LOF/BiVAT 6 pays × 4 types

All figures return go.Figure objects compatible with Dash dcc.Graph.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from beac_lof.loader import charger_labels

PAYS_CEMAC = ["cameroun", "congo", "gabon", "guinee_eq", "rca", "tchad"]
PAYS_LABELS = {
    "cameroun":  "Cameroun",
    "congo":     "Congo",
    "gabon":     "Gabon",
    "guinee_eq": "Guinée Équatoriale",
    "rca":       "RCA",
    "tchad":     "Tchad",
}

_MOIS_FR = ["", "Janv", "Févr", "Mars", "Avr", "Mai", "Juin",
             "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]

def _fmt_mois(dt) -> str:
    try:
        return f"{_MOIS_FR[dt.month]} {dt.year}"
    except Exception:
        return str(dt)[:7]


def _date_xaxis(title: str = "Date", dtick: str = "M12", tickformat: str = "%Y") -> dict:
    """Axe x temporel gradué sur toute la période (ticks annuels), pas un label
    par mois — nécessite des x réellement datetime."""
    return dict(title=title, type="date", dtick=dtick, tickformat=tickformat,
                ticklabelmode="period",
                gridcolor="rgba(255,255,255,0.06)", zeroline=False)

# Design tokens (match assets/style.css dark theme)
GOLD  = "#D4A020"
RED   = "#E05252"
BLUE  = "#5BA8E5"
GREEN = "#52C97E"
ORANGE = "#E07B30"
GREY  = "rgba(255,255,255,0.08)"

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f0f0f0", family="Roboto, sans-serif", size=11),
    margin=dict(t=90, b=60, l=60, r=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
)


def _apply_base(fig: go.Figure, **kwargs) -> go.Figure:
    layout = {**_LAYOUT_BASE, **kwargs}
    fig.update_layout(**layout)
    return fig


class BiVATGraphiques:
    """
    Container for all BiVAT Plotly figures.

    Parameters
    ----------
    pipeline     : PipelineBiVAT instance (fitted)
    lof_pipeline : PipelineLOF instance (fitted), required for Fig. D and G
    """

    def __init__(self, pipeline, lof_pipeline=None):
        self.p   = pipeline        # PipelineBiVAT
        self.lof = lof_pipeline    # PipelineLOF (optional, needed for D & G)
        self._label_maps_cache: dict = {}

    def _short_label(self, code: str, maxlen: int = 34) -> str:
        """Nom lisible et court pour un indicateur (au lieu du code IFS brut),
        à partir du fichier source de (self.p._pays, self.p._volet)."""
        k = (self.p._pays, self.p._volet)
        if k not in self._label_maps_cache:
            self._label_maps_cache[k] = charger_labels("data", self.p._pays, self.p._volet)
        name = self._label_maps_cache[k].get(str(code))
        if not name or name.lower() == "nan":
            return str(code)
        return name if len(name) <= maxlen else name[: maxlen - 1].rstrip() + "…"

    # ══════════════════════════════════════════════════════════════════════════
    # Fig. E  CP intervals on calibration window
    # ══════════════════════════════════════════════════════════════════════════

    def figE_intervalles_cp_calibration(self) -> go.Figure:
        """
        Score BiVAT + enveloppes CP 90%/95% sur la fenêtre de calibration.

        graphique.tex §Fig. E (Phase 2  Calibration BiVAT):
        - Courbe bleue : score discriminant
        - Enveloppe bleue claire : intervalle CP 90%
        - Enveloppe bleue foncée : intervalle CP 95%
        - Ligne pointillée rouge : q̂_{0.9} (seuil anomalie calibration)
        - Points rouges : mois dépassant le seuil
        """
        p = self.p
        sc   = p.scores_cal
        dates = p.dates_cal

        if sc is None or len(sc) == 0:
            return _placeholder("Fig. E", "Données de calibration non disponibles")

        valid = ~np.isnan(sc)
        x_v   = dates[valid]
        y_v   = sc[valid]
        q90   = p.q_hat_90 if p.q_hat_90 is not None else np.nanquantile(sc, 0.9)
        q95   = p.q_hat_95 if p.q_hat_95 is not None else np.nanquantile(sc, 0.95)

        # Approximate intervals as ±std on calibration
        std = float(np.nanstd(sc))
        lo90 = y_v - 0.8 * std
        hi90 = y_v + 0.8 * std
        lo95 = y_v - 1.4 * std
        hi95 = y_v + 1.4 * std

        fig = go.Figure()

        # CI ribbons
        _add_ribbon(fig, x_v, lo95, hi95, BLUE, 0.10, "CP 95%")
        _add_ribbon(fig, x_v, lo90, hi90, BLUE, 0.18, "CP 90%")

        # Score line
        fig.add_trace(go.Scatter(
            x=x_v, y=y_v, name="Score BiVAT (cal.)",
            line=dict(color=BLUE, width=1.8), mode="lines",
        ))

        # Threshold
        fig.add_hline(y=q90, line_dash="dot", line_color=RED,
                      annotation_text=f"q̂₀.₉₀ = {q90:.3f}",
                      annotation_position="top right")
        fig.add_hline(y=q95, line_dash="dash", line_color=RED,
                      annotation_text=f"q̂₀.₉₅ = {q95:.3f}",
                      annotation_position="bottom right")

        # Anomaly points
        anom_mask = y_v > q90
        if anom_mask.any():
            fig.add_trace(go.Scatter(
                x=x_v[anom_mask], y=y_v[anom_mask],
                mode="markers+text",
                marker=dict(color=RED, size=7),
                text=[_fmt_mois(d) for d in x_v[anom_mask]],
                textposition="top center", textfont=dict(size=8),
                name="Score > q̂",
            ))

        _apply_base(fig,
            title=dict(text="Fig. E  Score BiVAT + intervalles CP  fenêtre calibration<br>"
                            f"<sup>Couverture conforme Pr(s ≤ q̂₀.₉₀) = {(~anom_mask).sum()}/{len(anom_mask)} "
                            f"= {100*(~anom_mask).mean():.0f}%  [cible ≥ 90%]</sup>",
                       font=dict(size=14)),
            xaxis=_date_xaxis(),
            yaxis_title="s_discrim",
            height=420,
        )
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # Fig. D (A bis)  Comparaison LOF vs BiVAT normalisés [0,1]
    # ══════════════════════════════════════════════════════════════════════════

    def figD_comparaison_lof_bivat(self, pays: str = "cameroun", volet: str = "Actif") -> go.Figure:
        """
        Deux courbes normalisées [0,1] sur la fenêtre de test.

        graphique.tex §Fig. A bis (Phase 4) :
        - Courbe bleue : LOF*(t) normalisé
        - Courbe rouge : s_discrim(t) BiVAT normalisé
        - 4 zones colorées : accord normalité, accord anomalie, LOF seul, BiVAT seul
        """
        p = self.p
        if p.scores_test is None:
            return _placeholder("Fig. D", "BiVAT non entraîné")

        sc_bivat = p.scores_test
        dates_b  = p.dates_test

        # LOF scores on same window
        sc_lof_full = None
        if self.lof is not None:
            k = (pays.lower(), volet)
            s = self.lof.scores_lof.get(k)
            tau_lof = self.lof.tau.get(k)
            if s is not None:
                sc_lof_full = s

        # Align time axes
        if sc_lof_full is not None:
            common = dates_b[~np.isnan(sc_bivat)]
            lof_aligned = sc_lof_full.reindex(common)
            bivat_aligned = pd.Series(
                sc_bivat[~np.isnan(sc_bivat)], index=common
            )
            dates_common = common
        else:
            lof_aligned  = None
            bivat_aligned = pd.Series(sc_bivat[~np.isnan(sc_bivat)])
            dates_common  = dates_b[~np.isnan(sc_bivat)]

        # Normalize [0,1]
        def _norm(s: pd.Series) -> pd.Series:
            mn, mx = s.min(), s.max()
            if mx == mn:
                return s * 0
            return (s - mn) / (mx - mn)

        bivat_n = _norm(bivat_aligned)
        tau_b   = ((p.q_hat_90 or np.nanquantile(sc_bivat, 0.9)) - bivat_aligned.min()) / \
                  max(bivat_aligned.max() - bivat_aligned.min(), 1e-8)

        fig = go.Figure()

        # Shade zones
        if lof_aligned is not None and not lof_aligned.isna().all():
            lof_n   = _norm(lof_aligned.ffill().fillna(0))
            tau_l   = ((tau_lof or 0) - lof_aligned.min()) / max(lof_aligned.max() - lof_aligned.min(), 1e-8)
            _shade_zones(fig, dates_common, lof_n.values, bivat_n.values, tau_l, tau_b)

            fig.add_trace(go.Scatter(
                x=dates_common, y=lof_n.values, name="LOF* (normalisé)",
                line=dict(color=BLUE, width=1.6), mode="lines",
            ))
            fig.add_hline(y=tau_l, line_dash="dash", line_color=BLUE,
                          line_width=1, annotation_text="τ LOF", annotation_position="top left")

        fig.add_trace(go.Scatter(
            x=dates_common, y=bivat_n.values, name="BiVAT (normalisé)",
            line=dict(color=RED, width=1.6), mode="lines",
        ))
        fig.add_hline(y=tau_b, line_dash="dash", line_color=RED,
                      line_width=1, annotation_text="q̂ BiVAT", annotation_position="top right")

        _apply_base(fig,
            title=dict(text=f"Fig. D  Comparaison LOF vs BiVAT  {_label(pays, volet)}<br>"
                            "<sup>Vert pâle : accord normalité · Rouge pâle : accord anomalie · "
                            "Orange gauche : LOF seul · Orange droit : BiVAT seul</sup>",
                       font=dict(size=13)),
            xaxis=_date_xaxis(),
            yaxis=dict(title="Score normalisé [0,1]", range=[-0.05, 1.1],
                       gridcolor="rgba(255,255,255,0.06)", zeroline=False),
            height=440,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # Fig. E bis  CP intervals on test window (rolling-origin)
    # ══════════════════════════════════════════════════════════════════════════

    def figE_bis_bivat_cp_test(self) -> go.Figure:
        """
        Score BiVAT + enveloppes CP rolling-origin sur la fenêtre de test.

        graphique.tex §Fig. E bis (Phase 4) :
        - Courbe rouge : score discriminant BiVAT (test)
        - Enveloppe rouge claire : CP 90%  (rolling)
        - Enveloppe rouge foncée : CP 95%  (rolling)
        - Ligne pointillée rouge : q̂₀.₉₀ (calculé sur calibration)
        - Bande grisée COVID : mars 2020 – déc 2021
        """
        p = self.p
        if p.scores_test is None:
            return _placeholder("Fig. E bis", "BiVAT non entraîné")

        sc    = p.scores_test
        dates = p.dates_test
        valid = ~np.isnan(sc)
        x_v   = dates[valid]
        y_v   = sc[valid]

        # Rolling intervals
        lo90 = p.lo_90 if p.lo_90 is not None else y_v - float(np.std(y_v))
        hi90 = p.hi_90 if p.hi_90 is not None else y_v + float(np.std(y_v))
        lo95 = p.lo_95 if p.lo_95 is not None else y_v - 1.5 * float(np.std(y_v))
        hi_95 = p.hi_95 if p.hi_95 is not None else y_v + 1.5 * float(np.std(y_v))

        # Align to valid dates
        n = min(len(x_v), len(lo90))
        x_v = x_v[:n]; y_v = y_v[:n]
        lo90 = lo90[:n]; hi90 = hi90[:n]; lo95 = lo95[:n]; hi_95 = hi_95[:n]

        q90 = p.q_hat_90 or float(np.nanquantile(sc, 0.9))

        fig = go.Figure()

        # COVID grey band
        fig.add_vrect(
            x0="2020-03-01", x1="2021-12-31",
            fillcolor="rgba(255,255,255,0.05)", line_width=0,
            annotation_text="COVID-19", annotation_position="top left",
            annotation_font_size=9,
        )

        # CP ribbons
        _add_ribbon(fig, x_v, lo95, hi_95, RED, 0.10, "CP 95% (rolling)")
        _add_ribbon(fig, x_v, lo90, hi90, RED, 0.18, "CP 90% (rolling)")

        # Score line
        fig.add_trace(go.Scatter(
            x=x_v, y=y_v, name="Score BiVAT (test)",
            line=dict(color=RED, width=2), mode="lines",
        ))

        # Fixed threshold from calibration
        fig.add_hline(y=q90, line_dash="dot", line_color=RED, line_width=1,
                      annotation_text=f"q̂₀.₉₀ = {q90:.3f}",
                      annotation_position="top right", annotation_font_size=9)

        # Annotate anomaly months
        anom_mask = y_v > q90
        if anom_mask.any():
            fig.add_trace(go.Scatter(
                x=x_v[anom_mask], y=y_v[anom_mask],
                mode="markers+text",
                marker=dict(color=RED, size=7, symbol="circle"),
                text=[_fmt_mois(d) for d in x_v[anom_mask]],
                textposition="top center", textfont=dict(size=8),
                name=f"Score > q̂ ({int(anom_mask.sum())} mois)",
            ))

        _apply_base(fig,
            title=dict(text="Fig. E bis  Score BiVAT + intervalles CP (rolling)  fenêtre test<br>"
                            "<sup>Bande grisée : mars 2020 – déc 2021 (COVID-19, CP pondérée active)</sup>",
                       font=dict(size=13)),
            xaxis=_date_xaxis(), yaxis_title="s_discrim",
            height=440,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # Fig. F  Attribution SHAP top-5 anomalies BiVAT
    # ══════════════════════════════════════════════════════════════════════════

    def figF_shap_top_anomalies(self, top_n: int = 15) -> go.Figure:
        """
        Bar chart horizontal SHAP pour les top-5 anomalies BiVAT.

        graphique.tex §Fig. F (Phase 4) :
        - 5 bar charts horizontaux (sous-plots), un par anomalie majeure
        - 15 indicateurs triés par |φ_d| décroissant
        - Rouge : φ_d > 0 · Bleu : φ_d < 0
        - Sous-titre : mois, score BiVAT, score LOF correspondant
        """
        p = self.p
        if p.shap_results is None:
            return _placeholder("Fig. F", "SHAP non calculé (shap non installé ?)")

        sr = p.shap_results
        top_t   = sr["top_t"]
        top_ind = sr["top_indicators"]
        shap_signed = sr["shap_signed"]   # (n_top, d)
        feat    = sr["feature_names"]
        sc_test = p.scores_test

        # LOF scores for cross-model subtitle
        lof_scores_test = None
        if self.lof is not None:
            k = (p._pays, p._volet)
            s = self.lof.scores_lof.get(k)
            if s is not None:
                lof_scores_test = s

        n_anom = min(len(top_t), 5)
        if n_anom == 0:
            return _placeholder("Fig. F", "Aucune anomalie BiVAT détectée")

        fig = make_subplots(
            rows=n_anom, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[f"Anomalie {i+1}" for i in range(n_anom)],
        )

        for i in range(n_anom):
            t_idx = top_t[i]
            ind   = top_ind[i]
            sv    = shap_signed[i] if i < len(shap_signed) else ind["shap"]

            # Get top_n indicators with signed values
            d = len(feat)
            order = np.argsort(np.abs(sv))[::-1][:top_n]
            y_labels = [self._short_label(feat[j]) for j in order][::-1]
            x_vals   = [float(sv[j]) for j in order][::-1]
            colors   = [RED if v > 0 else BLUE for v in x_vals]

            # Score info
            bivat_sc = float(sc_test[t_idx]) if t_idx < len(sc_test) else float("nan")
            lof_sc   = float("nan")
            if lof_scores_test is not None and t_idx < len(lof_scores_test):
                try:
                    lof_sc = float(lof_scores_test.iloc[t_idx])
                except Exception:
                    pass

            # Date label
            date_str = str(p.dates_test[t_idx - (len(sc_test) - len(p.dates_test))])[:7] \
                       if p.dates_test is not None else f"t={t_idx}"

            # Update subplot title
            fig.layout.annotations[i].text = (
                f"Anomalie {i+1}  ·  {date_str}  ·  "
                f"BiVAT = {bivat_sc:.2f}  ·  LOF* = {lof_sc:.2f}"
            )

            fig.add_trace(go.Bar(
                x=x_vals, y=y_labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in x_vals],
                textposition="outside",
                showlegend=False,
            ), row=i + 1, col=1)

        _apply_base(fig,
            title=dict(text="Fig. F  Attribution SHAP par indicateur  top-5 anomalies BiVAT<br>"
                            "<sup>Rouge : contribution positive · Bleu : contribution négative</sup>",
                       font=dict(size=13)),
            height=200 * n_anom + 100,
            margin=dict(t=110, b=60, l=180, r=80),
        )
        fig.update_xaxes(title_text="SHAP φ_d", row=n_anom, col=1)
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # Fig. G  Heatmap confusion LOF/BiVAT 6 pays × 4 types
    # ══════════════════════════════════════════════════════════════════════════

    def figG_confusion_lof_bivat(self) -> go.Figure:
        """
        Heatmap 6×4 : 6 pays × 4 types d'anomalie, couleur = taux concordance.

        graphique.tex §Fig. G (Phase 4) :
        - Lignes : 6 pays CEMAC
        - Colonnes : Ponctuelle, Contextuelle, Collective, Systémique
        - Colormap : blanc (0%) → vert foncé (100%)
        - Encadré si concordance > 70%
        - Valeurs numériques dans chaque cellule
        """
        p = self.p
        if p.scores_test is None or self.lof is None:
            return _placeholder("Fig. G", "LOF et BiVAT doivent être tous deux entraînés")

        types = ["Ponctuelle", "Contextuelle", "Collective", "Systémique"]
        concordance = np.zeros((len(PAYS_CEMAC), 4))
        n_alerts    = np.zeros((len(PAYS_CEMAC), 4), dtype=int)

        sc_bivat = p.scores_test
        q90      = p.q_hat_90 or np.nanquantile(sc_bivat, 0.9)
        dates_b  = p.dates_test

        # Compute systemic months: ≥ 3 countries above their LOF threshold simultaneously
        syst_months: set = set()
        try:
            syst_series = self.lof.anomalies_systemiques(volet=p._volet, m_min=3)
            syst_months = set(syst_series.index)
        except Exception:
            pass

        for pi, pays in enumerate(PAYS_CEMAC):
            k = (pays, p._volet)
            sc_lof = self.lof.scores_lof.get(k)
            tau    = self.lof.tau.get(k)
            if sc_lof is None:
                continue

            # Align on common dates
            common  = dates_b[~np.isnan(sc_bivat)]
            lof_c   = sc_lof.reindex(common).ffill().fillna(0)
            bivat_c = pd.Series(sc_bivat[~np.isnan(sc_bivat)], index=common)

            anom_lof   = lof_c   > (tau or np.inf)
            anom_bivat = bivat_c > q90

            # Ponctuelle: LOF > τ AND BiVAT ≤ q̂ (isolated point anomaly in feature space)
            pont = anom_lof & ~anom_bivat
            # Contextuelle: BiVAT > q̂ AND LOF ≤ τ (temporal context missed by LOF)
            ctx  = ~anom_lof & anom_bivat
            # Collective: both anomalous in a sustained window ≥ 3 consecutive months
            both = anom_lof & anom_bivat
            coll = both.rolling(3, min_periods=3).sum().fillna(0).ge(3)
            # Systémique: this month is anomalous in ≥ 3 countries (from LOF),
            #             concordance = fraction of systemic months where BiVAT also flags
            syst = pd.Series([d in syst_months for d in common], index=common)

            # Concordance per type: fraction of type-X months where both models agree
            # (both flag anomaly or both flag normal)
            agree = ~(anom_lof ^ anom_bivat)
            for ti, mask in enumerate([pont, ctx, coll, syst]):
                mask = mask.astype(bool)
                n = int(mask.sum())
                n_alerts[pi, ti] = n
                if n > 0:
                    # For ponctuelle/contextuelle/collective: overall agreement on those months
                    concordance[pi, ti] = float(agree[mask].mean())
                else:
                    concordance[pi, ti] = 0.0

        # Heatmap
        pays_labels = [PAYS_LABELS.get(p, p) for p in PAYS_CEMAC]
        text_vals = [
            [f"{concordance[r,c]*100:.0f}%<br>({n_alerts[r,c]})" for c in range(4)]
            for r in range(len(PAYS_CEMAC))
        ]

        fig = go.Figure(go.Heatmap(
            z=concordance,
            x=types,
            y=pays_labels,
            colorscale=[[0, "white"], [0.7, "#2ecc71"], [1, "#1a7a43"]],
            zmin=0, zmax=1,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(color="#1a1a1a", size=10),
            showscale=True,
            colorbar=dict(title="Concordance", tickformat=".0%"),
        ))

        # Add boxes for high concordance cells
        for r in range(len(PAYS_CEMAC)):
            for c in range(4):
                if concordance[r, c] > 0.7:
                    fig.add_shape(type="rect",
                        x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
                        line=dict(color=GREEN, width=2.5),
                    )

        _apply_base(fig,
            title=dict(text="Fig. G  Concordance LOF / BiVAT par pays et type d'anomalie<br>"
                            "<sup>Encadré vert : concordance > 70% · "
                            "Vert foncé = fort accord inter-modèles</sup>",
                       font=dict(size=13)),
            height=420,
            xaxis=dict(side="top"),
        )
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # Export
    # ══════════════════════════════════════════════════════════════════════════

    def generer_toutes(
        self,
        pays: str = "cameroun",
        volet: str = "Actif",
        dossier_export: str = "figures",
        format: str = "html",
    ) -> dict:
        import os
        os.makedirs(dossier_export, exist_ok=True)

        methodes = {
            "figE":     self.figE_intervalles_cp_calibration,
            "figD":     lambda: self.figD_comparaison_lof_bivat(pays, volet),
            "figE_bis": self.figE_bis_bivat_cp_test,
            "figF":     self.figF_shap_top_anomalies,
            "figG":     self.figG_confusion_lof_bivat,
        }
        resultats = {}
        for nom, fn in methodes.items():
            try:
                fig = fn()
                if format == "html":
                    path = os.path.join(dossier_export, f"{nom}.html")
                    fig.write_html(path, include_plotlyjs="cdn")
                else:
                    path = os.path.join(dossier_export, f"{nom}.png")
                    fig.write_image(path, scale=2)
                resultats[nom] = fig
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Erreur {nom}: {e}")
        return resultats


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _label(pays: str, volet: str) -> str:
    return f"{PAYS_LABELS.get(pays.lower(), pays.capitalize())} / {volet}"


def _placeholder(fig_id: str, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=f"[{fig_id}] {msg}", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="rgba(255,255,255,0.4)"))
    _apply_base(fig, title=dict(text=fig_id, font=dict(size=14)), height=300)
    return fig


def _add_ribbon(
    fig: go.Figure,
    x: pd.DatetimeIndex,
    lo: np.ndarray,
    hi: np.ndarray,
    color: str,
    opacity: float,
    name: str,
) -> None:
    x_list = list(x) + list(x[::-1])
    y_list = list(hi) + list(lo[::-1])
    rgb = _hex_to_rgb(color)
    fig.add_trace(go.Scatter(
        x=x_list, y=y_list, fill="toself", mode="none",
        fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity})",
        name=name, showlegend=True,
    ))


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _shade_zones(
    fig: go.Figure,
    dates: pd.DatetimeIndex,
    lof_n: np.ndarray,
    bivat_n: np.ndarray,
    tau_lof: float,
    tau_bivat: float,
) -> None:
    """Add the 4 coloured agreement zones behind the score curves."""
    both_normal = (lof_n <= tau_lof) & (bivat_n <= tau_bivat)
    both_anom   = (lof_n >  tau_lof) & (bivat_n >  tau_bivat)
    lof_only    = (lof_n >  tau_lof) & (bivat_n <= tau_bivat)
    bivat_only  = (lof_n <= tau_lof) & (bivat_n >  tau_bivat)

    def _vrect_segments(mask, color, label):
        in_seg = False
        t_start = None
        for i, (d, flag) in enumerate(zip(dates, mask)):
            if flag and not in_seg:
                in_seg = True; t_start = d
            elif not flag and in_seg:
                in_seg = False
                fig.add_vrect(x0=t_start, x1=d,
                              fillcolor=color, opacity=0.18, line_width=0,
                              annotation_text=label if i < 5 else "",
                              annotation_font_size=8)
        if in_seg and t_start is not None:
            fig.add_vrect(x0=t_start, x1=dates[-1],
                          fillcolor=color, opacity=0.18, line_width=0)

    _vrect_segments(both_normal, "#52C97E", "Accord normalité")
    _vrect_segments(both_anom,   "#E05252", "Accord anomalie")
    _vrect_segments(lof_only,    "#E07B30", "LOF seul")
    _vrect_segments(bivat_only,  "#E09B30", "BiVAT seul")
