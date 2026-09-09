"""Callbacks page Données — sélection fichier + prévisualisation + suppression."""
import os
import dash
from dash import callback, Output, Input, State, ctx, html

from pages.donnees import (
    scan_files, _build_file_rows, _read_file_meta, _read_file_full,
    _preview_table_from_df, _stats_panel, _stat_cards, invalidate_scan_cache,
)


@callback(
    Output("preview-table-container", "children"),
    Output("file-list-container",     "children"),
    Output("file-info-bar",           "children"),
    Output("donnees-stats-container", "children"),
    Output("donnees-stat-cards",      "children"),
    Input({"type": "file-row", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_file_select(clicks):
    triggered = ctx.triggered_id
    if triggered is None or not any(c for c in (clicks or []) if c):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    idx   = triggered["index"]
    files = scan_files()

    if idx >= len(files) or not files[idx].get("ok"):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    path = files[idx].get("path", "")
    meta = _read_file_meta(path)

    new_rows = _build_file_rows(files, selected_idx=idx)
    preview  = _preview_table_from_df(meta.get("df"))
    stats    = _stats_panel(meta.get("df"))
    cards    = _stat_cards(meta)

    n_obs = meta.get("n_obs", "—")
    n_ind = meta.get("n_ind", "—")
    name  = files[idx]["name"]

    info_bar = [
        html.Span("Sélectionné", className="tag tag-gold"),
        html.Span(name, style={"fontSize": "12px", "fontWeight": "600",
                               "overflow": "hidden", "textOverflow": "ellipsis",
                               "whiteSpace": "nowrap", "maxWidth": "300px"}),
        html.Span(f"{n_obs} périodes · {n_ind} indicateurs",
                  style={"marginLeft": "auto", "fontSize": "10px",
                         "color": "var(--muted)", "whiteSpace": "nowrap"}),
    ]

    return preview, new_rows, info_bar, stats, cards


@callback(
    Output("btn-open-folder", "title"),
    Input("btn-open-folder",  "n_clicks"),
    prevent_initial_call=True,
)
def open_data_folder(n):
    if n:
        data_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data"))
        os.makedirs(data_dir, exist_ok=True)
        try:
            os.startfile(data_dir)
        except Exception:
            pass
    return "Dossier ouvert"


@callback(
    Output("dl-export-csv", "data"),
    Input("btn-export-csv",  "n_clicks"),
    State({"type": "file-row", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def export_csv(n, row_clicks):
    if not n:
        return dash.no_update

    files = scan_files()
    sel_idx = 0
    if row_clicks:
        for i, c in reversed(list(enumerate(row_clicks))):
            if c:
                sel_idx = i
                break

    if sel_idx >= len(files) or not files[sel_idx].get("ok"):
        return dash.no_update

    try:
        from dash import dcc
        df = _read_file_full(files[sel_idx]["path"])
        return dcc.send_data_frame(df.to_csv,
                                   filename=files[sel_idx]["name"].replace(".xlsx", ".csv"))
    except Exception:
        return dash.no_update


@callback(
    Output("app-modal-overlay",     "className"),
    Output("app-modal-title",       "children"),
    Output("app-modal-body",        "children"),
    Output("app-modal-footer",      "children"),
    Output("store-donnees-del-idx", "data"),
    Input({"type": "file-del", "index": dash.ALL}, "n_clicks"),
    State({"type": "file-row", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def ask_delete_file(clicks, row_ids):
    triggered = ctx.triggered_id
    if triggered is None or not any(c for c in (clicks or []) if c):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    idx   = triggered["index"]
    files = scan_files()
    fname = files[idx]["name"] if idx < len(files) else "ce fichier"

    body = html.Div(
        f"Supprimer définitivement « {fname} » du dossier data/ ? "
        "Cette action est irréversible.",
        style={"lineHeight": "1.5"},
    )
    footer = [
        html.Button("Annuler", id="app-modal-cancel", className="btn btn-ghost", n_clicks=0),
        html.Button("Supprimer", id="app-modal-confirm-btn-delfile",
                    className="btn btn-red", n_clicks=0),
    ]
    return "open", "Supprimer le fichier", body, footer, idx


@callback(
    Output("file-list-container", "children", allow_duplicate=True),
    Output("app-modal-overlay",   "className", allow_duplicate=True),
    Input("app-modal-confirm-btn-delfile", "n_clicks"),
    State("store-donnees-del-idx", "data"),
    prevent_initial_call=True,
)
def do_delete_file(n, del_idx):
    if not n or del_idx is None:
        return dash.no_update, dash.no_update
    files = scan_files()
    if del_idx < len(files) and files[del_idx].get("ok"):
        try:
            os.remove(files[del_idx]["path"])
        except Exception:
            pass
    invalidate_scan_cache()
    return _build_file_rows(scan_files(), selected_idx=0), ""
