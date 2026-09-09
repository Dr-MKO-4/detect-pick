"""Callbacks secondaires : rapport HTML, recalibrage LOF, paramètres."""
import os
import hashlib
import base64
import datetime
import dash
from dash import callback, clientside_callback, Output, Input, State, ctx, html
from callbacks.pipeline import _pipeline_state
from database import models as _m


# ── Génération rapport HTML ────────────────────────────────────────────────────

@callback(
    Output("dl-rapport",            "data"),
    Output("rapport-status",        "children"),
    Output("rapport-history-count", "children"),
    Output("rapport-history-list",  "children"),
    Output("rapport-last-gen",      "children"),
    Input("btn-generate-rapport",   "n_clicks"),
    State("dd-rapport-type",    "value"),
    State("dd-rapport-pays",    "value"),
    State("dd-rapport-modele",  "value"),
    State("chk-rapport-format", "value"),
    State("store-auth",         "data"),
    prevent_initial_call=True,
)
def generate_rapport(n, rtype, rpays, rmodele, formats, auth):
    if not n:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Le rapport doit refléter le périmètre choisi (pays/modèle) — on cherche
    # d'abord le run correspondant en base, avec repli sur le dernier run en
    # mémoire (comportement historique) si aucun run DB ne correspond.
    db_run = _m.get_latest_run(pays=rpays, modele=rmodele)
    results = _pipeline_state.get("results")
    if db_run:
        pays_label  = db_run.get("pays", "inconnu")
        volet_label = db_run.get("volet", "")
        run_id      = db_run.get("id")
        n_obs_real  = "—"
        n_anom      = db_run.get("n_anomalies", "—")
        tau_val     = db_run.get("tau_mean", "N/A")
    elif results is not None:
        pays_label  = results.get("pays",        "inconnu")
        volet_label = results.get("volet",       "")
        run_id      = results.get("run_id")
        n_obs_real  = results.get("n_obs",       "—")
        n_anom      = results.get("n_anomalies", "—")
        tau_val     = results.get("tau_mean",    "N/A")
    else:
        return (dash.no_update,
                "Aucune analyse disponible pour ce périmètre. Lancez d'abord l'analyse.",
                dash.no_update, dash.no_update, dash.no_update)

    now     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    user_id = (auth or {}).get("user_id")

    html_content = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>Rapport BEAC — {rtype}</title>
<style>
body {{font-family:Arial,sans-serif;margin:40px;color:#1a1a1a;}}
h1 {{color:#7a4f00;border-bottom:2px solid #9e6f1a;padding-bottom:8px;}}
h2 {{color:#7a4f00;margin-top:24px;}}
table {{border-collapse:collapse;width:100%;margin:16px 0;}}
th,td {{border:1px solid #ccc;padding:6px 10px;text-align:left;}}
th {{background:#f5f0e8;}}
.gold {{color:#9e6f1a;font-weight:700;}}
.meta {{font-size:11px;color:#888;}}
</style></head><body>
<h1>BANQUE DES ÉTATS DE L'AFRIQUE CENTRALE</h1>
<p class="meta">Rapport généré le {now} · Type : {rtype} · Modèle : {rmodele.upper()}</p>
<h2>Périmètre</h2>
<table><tr><th>Pays</th><th>Volet</th><th>Modèle</th><th>Observations</th></tr>
<tr><td>{pays_label.capitalize()}</td><td>{volet_label}</td>
    <td class="gold">{rmodele.upper()}</td><td>{n_obs_real}</td></tr>
</table>
<h2>Résultats LOF*</h2>
<p>Seuil τ = {tau_val} · Anomalies détectées : {n_anom}</p>
<p class="meta">Généré par BEAC Anomaly Detector v1.0</p>
</body></html>"""

    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    base_name = f"rapport_beac_{pays_label}_{volet_label}_{datetime.date.today()}"

    want_pdf = "pdf" in (formats or [])
    if want_pdf:
        fname = base_name + ".pdf"
        file_path_disk = os.path.join(reports_dir, fname)
        pdf_bytes = _build_pdf_report(rtype, rmodele, pays_label, volet_label,
                                      n_obs_real, tau_val, n_anom, now)
        with open(file_path_disk, "wb") as _f:
            _f.write(pdf_bytes)
        download = dict(content=base64.b64encode(pdf_bytes).decode("ascii"),
                        filename=fname, type="application/pdf", base64=True)
    else:
        fname = base_name + ".html"
        file_path_disk = os.path.join(reports_dir, fname)
        with open(file_path_disk, "w", encoding="utf-8") as _f:
            _f.write(html_content)
        download = dict(content=html_content, filename=fname, type="text/html")

    _m.save_report(run_id, user_id, rtype, file_path_disk,
                   {"rpays": rpays, "rmodele": rmodele, "formats": formats})
    _m.log_audit(user_id, "report_generated", details={"fname": fname, "type": rtype})

    from pages.rapport import _history_rows
    history = _m.get_reports(limit=20)
    n_hist  = len(history)

    return (
        download,
        f"Rapport enregistré : {file_path_disk}",
        f"{n_hist} rapport{'s' if n_hist != 1 else ''}",
        _history_rows(history),
        f"Dernier : {now}",
    )


def _build_pdf_report(rtype, rmodele, pays_label, volet_label,
                      n_obs_real, tau_val, n_anom, now) -> bytes:
    """Génère le PDF via reportlab — mêmes sections que le rapport HTML."""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=24 * mm, bottomMargin=20 * mm,
                            leftMargin=20 * mm, rightMargin=20 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("BeacTitle", parent=styles["Title"],
                                 textColor=colors.HexColor("#7a4f00"), fontSize=16)
    meta_style = ParagraphStyle("BeacMeta", parent=styles["Normal"],
                                fontSize=9, textColor=colors.grey)
    h2_style = ParagraphStyle("BeacH2", parent=styles["Heading2"],
                              textColor=colors.HexColor("#7a4f00"))

    story = [
        Paragraph("BANQUE DES ÉTATS DE L'AFRIQUE CENTRALE", title_style),
        Paragraph(f"Rapport généré le {now} · Type : {rtype} · Modèle : {rmodele.upper()}", meta_style),
        Spacer(1, 14),
        Paragraph("Périmètre", h2_style),
        Table(
            [["Pays", "Volet", "Modèle", "Observations"],
             [pays_label.capitalize(), volet_label, rmodele.upper(), str(n_obs_real)]],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f0e8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]),
        ),
        Spacer(1, 14),
        Paragraph("Résultats", h2_style),
        Paragraph(f"Seuil τ = {tau_val} · Anomalies détectées : {n_anom}", styles["Normal"]),
        Spacer(1, 20),
        Paragraph("Généré par BEAC Anomaly Detector v1.0", meta_style),
    ]
    doc.build(story)
    return buf.getvalue()


# ── Recalibrage LOF* (page Modèles) ──────────────────────────────────────────

@callback(
    Output("lof-calibration-status", "children"),
    Input("btn-run-lof-all",         "n_clicks"),
    State("dd-lof-tau",              "value"),
    prevent_initial_call=True,
)
def recalibrate_lof(n, tau_method):
    if not n:
        return dash.no_update
    lof_p = _pipeline_state.get("lof_pipeline")
    if lof_p is None:
        return [
            dash.html.Div("Aucun pipeline LOF chargé.",
                          style={"fontSize": "11px", "color": "var(--red)", "fontWeight": "700"}),
            dash.html.Div("Lancez d'abord l'analyse sur la page Analyse.",
                          style={"fontSize": "10px", "color": "var(--muted)"}),
        ]
    tau_labels = {"iqr15": "Q3 + 1,5 × IQR", "iqr20": "Q3 + 2,0 × IQR", "p95": "Percentile 95%"}
    return [
        dash.html.Div(f"Méthode de seuil : {tau_labels.get(tau_method, tau_method)}",
                      style={"fontSize": "11px", "color": "var(--green)", "fontWeight": "700"}),
        dash.html.Div("Relancez l'analyse pour recalculer.",
                      style={"fontSize": "10px", "color": "var(--muted)", "marginTop": "4px"}),
    ]


# ── Fermeture dropdowns menu Dash (clientside) ────────────────────────────────

clientside_callback(
    """
    function() {
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.menu-item-wrapper')) {
                document.querySelectorAll('.menu-dropdown').forEach(function(dd) {
                    dd.classList.add('hidden');
                });
            }
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("store-page", "data", allow_duplicate=True),
    Input("store-page",  "data"),
    prevent_initial_call=True,
)


# ── Thème via menubar ─────────────────────────────────────────────────────────

clientside_callback(
    """
    function(n, theme) {
        if (!n) return window.dash_clientside.no_update;
        var next = theme === 'dark' ? 'light' : 'dark';
        document.documentElement.dataset.theme = next;
        return next;
    }
    """,
    Output("store-theme", "data", allow_duplicate=True),
    Input("menu-theme-light", "n_clicks"),
    State("store-theme",      "data"),
    prevent_initial_call=True,
)

clientside_callback(
    """
    function(n, theme) {
        if (!n) return window.dash_clientside.no_update;
        document.documentElement.dataset.theme = 'dark';
        return 'dark';
    }
    """,
    Output("store-theme", "data", allow_duplicate=True),
    Input("menu-theme-dark", "n_clicks"),
    State("store-theme",     "data"),
    prevent_initial_call=True,
)


# ── Navigation catégories Paramètres ─────────────────────────────────────────
# Utilise callback_context.triggered pour identifier l'onglet cliqué,
# évitant le bug de comparaison de n_clicks quand plusieurs onglets ont été visités.

clientside_callback(
    """
    function(ng, na, nd, ns, nap) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length)
            return window.dash_clientside.no_update;

        var prop_id = ctx.triggered[0].prop_id;          // "settings-nav-general.n_clicks"
        var nav_id  = prop_id.split('.')[0];              // "settings-nav-general"
        var chosen  = nav_id.replace('settings-nav-', ''); // "general"

        var cats = ['general','apparence','donnees','securite','a-propos'];
        cats.forEach(function(c) {
            var el  = document.getElementById('settings-content-' + c);
            var nav = document.getElementById('settings-nav-' + c);
            var active = (c === chosen);
            if (el)  el.style.display = active ? '' : 'none';
            if (nav) {
                if (active) nav.classList.add('active');
                else        nav.classList.remove('active');
            }
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("store-page", "data", allow_duplicate=True),
    Input("settings-nav-general",   "n_clicks"),
    Input("settings-nav-apparence", "n_clicks"),
    Input("settings-nav-donnees",   "n_clicks"),
    Input("settings-nav-securite",  "n_clicks"),
    Input("settings-nav-a-propos",  "n_clicks"),
    prevent_initial_call=True,
)


# ── Thème via pills Paramètres > Apparence ────────────────────────────────────

clientside_callback(
    """
    function(nd, nl, theme) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length)
            return window.dash_clientside.no_update;
        var src = ctx.triggered[0].prop_id.split('.')[0];
        var next = (src === 'cfg-theme-dark') ? 'dark' : 'light';
        document.documentElement.dataset.theme = next;
        return next;
    }
    """,
    Output("store-theme", "data", allow_duplicate=True),
    Input("cfg-theme-dark",  "n_clicks"),
    Input("cfg-theme-light", "n_clicks"),
    State("store-theme",     "data"),
    prevent_initial_call=True,
)


# ── Ouvrir dossier data/ depuis Paramètres > Données ─────────────────────────

@callback(
    Output("cfg-data-path-display", "children"),
    Input("cfg-browse-data",        "n_clicks"),
    prevent_initial_call=True,
)
def cfg_browse_data(n):
    if not n:
        return dash.no_update
    data_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    try:
        os.startfile(data_dir)
    except Exception:
        pass
    return data_dir


# ── Changer le mot de passe ───────────────────────────────────────────────────

@callback(
    Output("cfg-pwd-feedback",  "children"),
    Output("cfg-pwd-current",   "value"),
    Output("cfg-pwd-new",       "value"),
    Output("cfg-pwd-confirm",   "value"),
    Input("cfg-btn-change-pwd", "n_clicks"),
    State("cfg-pwd-current",    "value"),
    State("cfg-pwd-new",        "value"),
    State("cfg-pwd-confirm",    "value"),
    State("store-auth",         "data"),
    prevent_initial_call=True,
)
def change_password(n, current, new_pwd, confirm, auth):
    _noup = dash.no_update
    if not n:
        return _noup, _noup, _noup, _noup

    username = (auth or {}).get("user", "")
    if not username:
        return _err("Non authentifié."), _noup, _noup, _noup

    if not current or not new_pwd or not confirm:
        return _err("Tous les champs sont obligatoires."), _noup, _noup, _noup

    if new_pwd != confirm:
        return _err("Les mots de passe ne correspondent pas."), _noup, "", ""

    if len(new_pwd) < 6:
        return _err("Le mot de passe doit comporter au moins 6 caractères."), _noup, _noup, _noup

    from database import models as _m
    user = _m.get_user(username)
    if not user:
        return _err("Utilisateur introuvable."), _noup, _noup, _noup

    current_hash = hashlib.sha256(current.encode()).hexdigest()
    if current_hash != user["password_hash"]:
        return _err("Mot de passe actuel incorrect."), "", _noup, _noup

    new_hash = hashlib.sha256(new_pwd.encode()).hexdigest()
    _m.change_password(username, new_hash)
    _m.log_audit(user["id"], "password_changed", details={})

    return _ok("Mot de passe mis à jour."), "", "", ""


def _err(msg: str):
    return dash.html.Span(msg, style={"color": "var(--red)", "fontSize": "11px"})


def _ok(msg: str):
    return dash.html.Span(msg, style={"color": "var(--green)", "fontSize": "11px"})


# ── Mise à jour DOM barre de progression (clientside, sans re-render page) ────

clientside_callback(
    """
    function(progress, label, running, bivatRunning) {
        var fill  = document.getElementById('pg-fill');
        var lbl   = document.getElementById('pg-label');
        var pct   = document.getElementById('pg-pct');
        var panel = document.getElementById('pg-running-panel');
        var btnRun  = document.getElementById('btn-run-pipeline');
        var btnStop = document.getElementById('btn-stop-pipeline');
        var anyRunning = Boolean(running) || Boolean(bivatRunning);

        if (fill) fill.style.width = (progress || 0) + '%';
        if (lbl)  lbl.textContent = label || 'En cours…';
        if (pct)  pct.textContent = (progress || 0) + '%';

        var p = progress || 0;
        var charge  = document.getElementById('pg-step-charge');
        var lofStep = document.getElementById('pg-step-lof');
        var figs    = document.getElementById('pg-step-figs');
        if (charge)  charge.className  = 'vs-step' + (p >= 20 ? ' vs-step-done' : p >= 5  ? ' vs-step-active' : '');
        if (lofStep) lofStep.className = 'vs-step' + (p >= 65 ? ' vs-step-done' : p >= 20 ? ' vs-step-active' : '');
        if (figs)    figs.className    = 'vs-step' + (p >= 95 ? ' vs-step-done' : p >= 65 ? ' vs-step-active' : '');

        if (panel)   panel.style.display   = anyRunning ? 'block' : 'none';
        if (btnRun)  btnRun.style.display  = anyRunning ? 'none'  : 'flex';
        if (btnStop) btnStop.style.display = anyRunning ? 'flex'  : 'none';

        return window.dash_clientside.no_update;
    }
    """,
    Output("pg-dummy", "children"),
    Input("store-progress",         "data"),
    Input("store-progress-label",   "data"),
    Input("store-pipeline-running", "data"),
    Input("store-bivat-running",    "data"),
)


# ── Animations globales persistantes (top bar + toast) ───────────────────────

clientside_callback(
    """
    function(running, bivatRunning, progress, label) {
        var anyRunning = Boolean(running) || Boolean(bivatRunning);

        var topBar  = document.getElementById('global-top-bar');
        var toast   = document.getElementById('global-status-toast');
        var title   = document.getElementById('gst-title');
        var lbl     = document.getElementById('gst-label');
        var pct     = document.getElementById('gst-pct');
        var fill    = document.getElementById('gst-bar-fill');

        if (topBar) topBar.style.display  = anyRunning ? 'block' : 'none';
        if (toast)  toast.style.display   = anyRunning ? 'block' : 'none';

        if (anyRunning) {
            var p = progress || 0;
            var model = bivatRunning ? 'BiVAT' : 'LOF*';
            if (title) title.textContent = model + '  —  ' + p + '%';
            if (lbl)   lbl.textContent   = label || 'Traitement en cours…';
            if (pct)   pct.textContent   = p + '%';
            if (fill)  fill.style.width  = p + '%';
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output("global-anim-dummy",      "children"),
    Input("store-pipeline-running",  "data"),
    Input("store-bivat-running",     "data"),
    Input("store-progress",          "data"),
    Input("store-progress-label",    "data"),
)


# ── Sélecteur d'année : zoom des figures temporelles sans recalcul ───────────
# Filtre uniquement la fenêtre affichée (xaxis.range) des figures à axe de
# type "date" — aucun impact sur le pipeline, qui continue de tourner sur
# tout l'historique (nécessaire pour STL/RPCA/LOF*).

clientside_callback(
    """
    function(annee) {
        var ids = ['graph-fig3', 'graph-fig10', 'graph-fig11', 'graph-figA',
                    'graph-figB', 'graph-figD', 'graph-figE', 'graph-figE_bis'];
        var range = (annee && annee !== 'all')
            ? [annee + '-01-01', (parseInt(annee, 10) + 1) + '-01-01']
            : null;
        ids.forEach(function(id) {
            var el = document.getElementById(id);
            if (!el || !window.Plotly) return;
            var gd = el.querySelector('.js-plotly-plot') || el;
            try {
                if (range) {
                    window.Plotly.relayout(gd, {'xaxis.range': range, 'xaxis.autorange': false});
                } else {
                    window.Plotly.relayout(gd, {'xaxis.autorange': true});
                }
            } catch (e) { /* figure pas encore montée : ignorer */ }
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("annee-dummy", "children"),
    Input("dd-annee", "value"),
    prevent_initial_call=True,
)


# ── Feedback visuel bouton Générer rapport ────────────────────────────────────

clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        var btn = document.getElementById('btn-generate-rapport');
        if (btn) {
            btn.disabled = true;
            btn.style.opacity = '0.6';
            var orig = btn.innerHTML;
            btn.textContent = 'Génération…';
            setTimeout(function() {
                btn.disabled = false;
                btn.style.opacity = '1';
                btn.innerHTML = orig;
            }, 4000);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("rapport-btn-dummy", "children"),
    Input("btn-generate-rapport", "n_clicks"),
    prevent_initial_call=True,
)


# ── Liaisons DOM globales, une seule fois (activation clavier, fermeture modal,
#    raccourci Ctrl+K) — regroupées en UN SEUL clientside_callback : Dash calcule
#    le hash `allow_duplicate` à partir des Input(s), donc plusieurs callbacks
#    distincts partageant le même unique Input("store-page","data") entreraient
#    en collision ("Duplicate callback outputs") s'ils restaient séparés.

clientside_callback(
    """
    function(_) {
        if (!window._beacGlobalBindingsDone) {
            // Activation clavier des éléments role="button" (Div non natifs) —
            // Enter/Espace déclenche le même .click() qu'un clic souris.
            document.addEventListener('keydown', function(e) {
                if (e.key !== 'Enter' && e.key !== ' ') return;
                var el = e.target.closest('[role="button"]');
                if (el) { e.preventDefault(); el.click(); }
            });

            // Modal générique : fermeture sur clic overlay / × / Annuler / Échap.
            document.addEventListener('click', function(e) {
                if (e.target && e.target.id === 'app-modal-overlay') {
                    e.target.className = '';
                }
                if (e.target.closest('#app-modal-close') || e.target.closest('#app-modal-cancel')) {
                    var ov = document.getElementById('app-modal-overlay');
                    if (ov) ov.className = '';
                }
            });
            document.addEventListener('keydown', function(e) {
                if (e.key !== 'Escape') return;
                var ov = document.getElementById('app-modal-overlay');
                if (ov) ov.className = '';
                var sp = document.getElementById('search-palette-overlay');
                if (sp) sp.classList.remove('open');
            });

            // Palette de recherche : Ctrl+K / Cmd+K ouvre, clic sur l'overlay ferme.
            document.addEventListener('keydown', function(e) {
                if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
                    e.preventDefault();
                    var ov = document.getElementById('search-palette-overlay');
                    if (ov) {
                        ov.classList.add('open');
                        setTimeout(function() {
                            var inp = document.getElementById('inp-search-query');
                            if (inp) inp.focus();
                        }, 30);
                    }
                }
            });
            document.addEventListener('click', function(e) {
                if (e.target && e.target.id === 'search-palette-overlay') {
                    e.target.classList.remove('open');
                }
            });

            window._beacGlobalBindingsDone = true;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("global-anim-dummy", "children", allow_duplicate=True),
    Input("store-page", "data"),
    prevent_initial_call=True,
)


# ── Notifications : ouverture/fermeture du panneau ────────────────────────────

clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        var panel = document.getElementById('notif-panel');
        if (panel) panel.classList.toggle('open');
        return window.dash_clientside.no_update;
    }
    """,
    Output("global-anim-dummy", "children", allow_duplicate=True),
    Input("btn-notif-bell", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("notif-panel-list", "children"),
    Output("store-notif-count", "data"),
    Input("btn-notif-bell",   "n_clicks"),
    Input("interval-notif",   "n_intervals"),
    State("store-auth",       "data"),
    prevent_initial_call=True,
)
def refresh_notifications(n_bell, n_interval, auth):
    user_id = (auth or {}).get("user_id")
    if not user_id:
        return dash.no_update, dash.no_update

    notifs = _m.get_notifications(user_id, limit=20)
    unread = _m.get_unread_count(user_id)

    if not notifs:
        items = [html.Div("Aucune notification.",
                          style={"padding": "16px", "fontSize": "11px", "color": "var(--muted2)"})]
    else:
        items = [
            html.Div(className="notif-item" + (" unread" if not nf["is_read"] else ""), children=[
                html.Div(nf["title"], className="notif-item-title"),
                html.Div(nf.get("body") or "", className="notif-item-body"),
                html.Div((nf.get("created_at") or "")[:16], className="notif-item-date"),
            ])
            for nf in notifs
        ]
        items.append(html.Div(
            html.Button("Tout marquer lu", id="btn-notif-mark-all", n_clicks=0,
                        className="btn btn-ghost btn-sm", style={"width": "100%", "margin": "8px"}),
        ))

    # Marquer comme lu à l'ouverture (déclenché par le clic sur la cloche, pas par le polling)
    if ctx.triggered_id == "btn-notif-bell":
        _m.mark_all_notifications_read(user_id)
        unread = 0

    return items, unread


@callback(
    Output("notif-panel-list", "children", allow_duplicate=True),
    Output("store-notif-count", "data", allow_duplicate=True),
    Input("btn-notif-mark-all", "n_clicks"),
    State("store-auth",         "data"),
    prevent_initial_call=True,
)
def mark_all_notifs_read(n, auth):
    if not n:
        return dash.no_update, dash.no_update
    user_id = (auth or {}).get("user_id")
    if user_id:
        _m.mark_all_notifications_read(user_id)
    return dash.no_update, 0


clientside_callback(
    """
    function(count) {
        var badge = document.getElementById('notif-badge');
        if (!badge) return window.dash_clientside.no_update;
        var n = count || 0;
        badge.textContent = n > 0 ? String(n) : '';
        badge.className = n > 0 ? 'notif-badge' : 'notif-badge hidden';
        return window.dash_clientside.no_update;
    }
    """,
    Output("global-anim-dummy", "children", allow_duplicate=True),
    Input("store-notif-count", "data"),
    prevent_initial_call=True,
)


# ── Ouverture de la palette de recherche via l'icône loupe de la sidebar ─────
# (le raccourci clavier Ctrl+K est lié dans le clientside_callback groupé ci-dessus)

clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        var ov = document.getElementById('search-palette-overlay');
        if (ov) {
            ov.classList.add('open');
            setTimeout(function() {
                var inp = document.getElementById('inp-search-query');
                if (inp) inp.focus();
            }, 30);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("global-anim-dummy", "children", allow_duplicate=True),
    Input("btn-open-search", "n_clicks"),
    prevent_initial_call=True,
)
