"""Chargement des fragments HTML d'aide et boîte de dialogue guide."""
import os

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTabWidget, QTextBrowser,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

_AIDE_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "aide")
_ICON     = os.path.join(os.path.dirname(__file__), "..", "assets", "logo_beac.jfif")

_CSS = """
    body { font-family: 'Segoe UI', Roboto, sans-serif; font-size: 13px;
           color: #1a1a1a; background: #fff; line-height: 1.5; }
    h2 { margin-top: 4px; }
    h3 { margin-top: 16px; color: #333; }
    table { border-collapse: collapse; width: 100%; }
    td { padding: 4px 8px; vertical-align: top; }
    kbd {
        background: #f0ede6; border: 1px solid #ccc; border-radius: 3px;
        padding: 1px 5px; font-family: 'Cascadia Code', 'Consolas', monospace;
        font-size: 11px;
    }
    code { background: #f0ede6; border-radius: 3px; padding: 1px 4px;
           font-family: 'Cascadia Code', Consolas, monospace; font-size: 11px; }
    li { margin-bottom: 4px; }
"""


def _load(filename: str) -> str:
    path = os.path.normpath(os.path.join(_AIDE_DIR, filename))
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"<p><i>Fichier introuvable : {filename}</i></p>"


def html_dialog(parent, title: str, sections: list[tuple[str, str]]) -> QDialog:
    from PyQt6.QtWidgets import QFrame
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(820, 640)
    if os.path.exists(_ICON):
        dlg.setWindowIcon(QIcon(_ICON))

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(0, 0, 0, 8)

    tabs = QTabWidget()
    tabs.setDocumentMode(True)
    tabs.setStyleSheet("""
        QTabWidget::pane { border: none; }
        QTabBar::tab { padding: 6px 18px; font-size: 12px;
                       font-family: 'Segoe UI', Roboto, sans-serif; }
        QTabBar::tab:selected { color: #7a4f00; border-bottom: 2px solid #D4A020; }
    """)

    for tab_name, html_content in sections:
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(f"<html><head><style>{_CSS}</style></head>"
                        f"<body style='padding:20px'>{html_content}</body></html>")
        browser.setFrameShape(QFrame.Shape.NoFrame)
        tabs.addTab(browser, tab_name)

    layout.addWidget(tabs)

    btns = QHBoxLayout()
    btns.addStretch()
    close_btn = QPushButton("Fermer")
    close_btn.setFixedWidth(100)
    close_btn.clicked.connect(dlg.accept)
    btns.addWidget(close_btn)
    btns.setContentsMargins(8, 0, 8, 0)
    layout.addLayout(btns)

    return dlg


def show_aide(parent) -> None:
    dlg = html_dialog(parent, "Guide d'utilisation BEAC Anomaly Detector", [
        ("Démarrage rapide", _load("general.html")),
        ("Pipeline LOF*",    _load("lof.html")),
        ("Pipeline BiVAT",   _load("bivat.html")),
        ("Guide des figures", _load("figures.html")),
        ("Interprétation",   _load("interpretation.html")),
        ("Raccourcis",       _load("raccourcis.html")),
    ])
    dlg.exec()


def show_about(parent) -> None:
    dlg = html_dialog(parent, "À propos BEAC Anomaly Detector", [
        ("À propos", _load("about.html")),
    ])
    dlg.exec()
