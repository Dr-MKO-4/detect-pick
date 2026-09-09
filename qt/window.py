"""Fenêtre principale BEAC (PyQt6 + QWebEngineView)."""
import os
import shutil
import ctypes
import ctypes.wintypes as _wt

from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import QUrl, Qt, QPoint, QTimer, QAbstractNativeEventFilter
from PyQt6.QtGui import QIcon


class _ResizeFilter(QAbstractNativeEventFilter):
    """Intercepte WM_NCHITTEST pour permettre le resize natif d'une fenêtre frameless."""
    _BORDER = 6  # largeur de la zone sensible en pixels

    # constantes NCHITTEST
    _HTLEFT, _HTRIGHT, _HTTOP = 10, 11, 12
    _HTTOPLEFT, _HTTOPRIGHT   = 13, 14
    _HTBOTTOM                  = 15
    _HTBOTTOMLEFT, _HTBOTTOMRIGHT = 16, 17

    def __init__(self, win: QMainWindow):
        super().__init__()
        self._win = win

    def nativeEventFilter(self, event_type, message):
        if event_type != b"windows_generic_MSG":
            return False, 0
        try:
            msg = ctypes.cast(int(message), ctypes.POINTER(_wt.MSG)).contents
            if msg.message != 0x0084:          # WM_NCHITTEST
                return False, 0
            w = self._win
            if w.isMaximized() or w.isFullScreen():
                return False, 0

            # Coordonnées écran → fenêtre
            x  = ctypes.c_short(msg.lParam & 0xFFFF).value
            y  = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
            gx, gy = w.x(), w.y()
            gw, gh = w.width(), w.height()
            cx, cy = x - gx, y - gy

            if cx < 0 or cy < 0 or cx > gw or cy > gh:
                return False, 0

            b  = self._BORDER
            ol = cx <= b
            or_ = cx >= gw - b
            ot = cy <= b
            ob = cy >= gh - b

            if   ot and ol:  hit = self._HTTOPLEFT
            elif ot and or_: hit = self._HTTOPRIGHT
            elif ob and ol:  hit = self._HTBOTTOMLEFT
            elif ob and or_: hit = self._HTBOTTOMRIGHT
            elif ol:         hit = self._HTLEFT
            elif or_:        hit = self._HTRIGHT
            elif ot:         hit = self._HTTOP
            elif ob:         hit = self._HTBOTTOM
            else:            return False, 0

            return True, hit
        except Exception:
            return False, 0

from .menubar import BEACMenuBar, MENUBAR_STYLE
from .aide import show_aide, show_about

_ICON = os.path.join(os.path.dirname(__file__), "..", "assets", "logo_beac.jfif")


class BEACWindow(QMainWindow):
    def __init__(self, url: str, dev: bool = False):
        super().__init__()
        self._url = url
        self._dev = dev
        self.setWindowTitle(
            "BEAC · Détection d'anomalies monétaires [DEV]" if dev else
            "BEAC · Détection d'anomalies monétaires  v1.0"
        )
        self.setMinimumSize(860, 520)
        self._full_size  = (1296, 720)
        self._login_size = (860, 520)
        self.resize(*self._login_size)

        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width()  - self._login_size[0]) // 2,
            (screen.height() - self._login_size[1]) // 2,
        )

        if os.path.exists(_ICON):
            self.setWindowIcon(QIcon(_ICON))

        self.webview = QWebEngineView(self)
        s = self.webview.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, True)
        self.setCentralWidget(self.webview)
        self.webview.load(QUrl(url))
        profile = self.webview.page().profile()
        profile.downloadRequested.connect(self._handle_download)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self._build_titlebar()
        self._build_menubar()
        self.menuBar().setVisible(False)

        # Resize natif via filtre WM_NCHITTEST (Windows uniquement)
        self._resize_filter = _ResizeFilter(self)
        QApplication.instance().installNativeEventFilter(self._resize_filter)
        self._current_page = "login"
        self._page_timer = QTimer(self)
        self._page_timer.setInterval(800)
        self._page_timer.timeout.connect(self._poll_page)
        self._page_timer.start()
        self.show()

    # ── Titlebar ──────────────────────────────────────────────────────────────

    def _build_titlebar(self):
        from PyQt6.QtWidgets import QToolBar, QSizePolicy as _SP

        class _TitleBar(QToolBar):
            _drag_pos = None

            def mousePressEvent(self, e):
                if e.button() == Qt.MouseButton.LeftButton:
                    self._drag_pos = (e.globalPosition().toPoint()
                                      - self.window().frameGeometry().topLeft())
                super().mousePressEvent(e)

            def mouseMoveEvent(self, e):
                if e.buttons() == Qt.MouseButton.LeftButton and self._drag_pos:
                    self.window().move(e.globalPosition().toPoint() - self._drag_pos)
                super().mouseMoveEvent(e)

            def mouseReleaseEvent(self, e):
                self._drag_pos = None
                super().mouseReleaseEvent(e)

            def mouseDoubleClickEvent(self, e):
                if e.button() == Qt.MouseButton.LeftButton:
                    w = self.window()
                    w.showNormal() if w.isMaximized() else w.showMaximized()
                super().mouseDoubleClickEvent(e)

        tb = _TitleBar("titlebar", self)
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setObjectName("custom-titlebar")
        tb.setStyleSheet("""
            QToolBar#custom-titlebar {
                background: #f0eeec;
                border-bottom: 1px solid rgba(32,31,29,.10);
                padding: 0 4px;
                spacing: 0;
                min-height: 28px;
                max-height: 28px;
            }
        """)

        spacer = QWidget()
        spacer.setSizePolicy(_SP.Policy.Expanding, _SP.Policy.Preferred)
        tb.addWidget(spacer)

        for sym, hover_bg, hover_fg, fn in [
            ("─", "#e0ddd6", "#111", self.showMinimized),
            ("□", "#e0ddd6", "#111", self._toggle_max),
            ("✕", "#E05252", "#fff", self.close),
        ]:
            btn = QPushButton(sym)
            btn.setFixedSize(30, 22)
            btn.setStyleSheet(
                f"QPushButton {{ border: none; border-radius: 0px;"
                f" color: #666; font-size: 12px; background: transparent; }}"
                f"QPushButton:hover {{ background: {hover_bg}; color: {hover_fg}; }}"
            )
            btn.clicked.connect(fn)
            tb.addWidget(btn)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self._custom_titlebar = tb

    # ── Menubar ───────────────────────────────────────────────────────────────

    def _build_menubar(self):
        mb = BEACMenuBar(self)
        self.setMenuBar(mb)
        mb.setStyleSheet(MENUBAR_STYLE)

        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 2, 4, 2)
        cl.setSpacing(1)
        for sym, hover_bg, hover_fg, fn in [
            ("─", "#e0ddd6", "#111", self.showMinimized),
            ("□", "#e0ddd6", "#111", self._toggle_max),
            ("✕", "#E05252", "#fff", self.close),
        ]:
            btn = QPushButton(sym)
            btn.setFixedSize(30, 22)
            btn.setStyleSheet(
                f"QPushButton {{ border: none; border-radius: 4px; color: #555;"
                f" font-size: 12px; background: transparent; }}"
                f"QPushButton:hover {{ background: {hover_bg}; color: {hover_fg}; }}"
            )
            btn.clicked.connect(fn)
            cl.addWidget(btn)
        mb.setCornerWidget(corner, Qt.Corner.TopRightCorner)

        # Fichier
        m = mb.addMenu("Fichier")
        self._add_action(m, "Importer données XLSX…", "Ctrl+O", self._import_xlsx)
        m.addSeparator()
        self._add_action(m, "Exporter rapport HTML",  "Ctrl+E",
                         lambda: self._nav("rapport"))
        self._add_action(m, "Exporter figures PNG…",  "Ctrl+Shift+E",
                         self._export_figures)
        m.addSeparator()
        self._add_action(m, "Quitter", "Ctrl+Q",
                         QApplication.instance().quit)

        # Affichage
        m = mb.addMenu("Affichage")
        for label, page, key in [
            ("Données", "donnees", "F1"),
            ("Analyse", "analyse", "F2"),
            ("Rapport", "rapport", "F3"),
            ("Modèles", "modeles", "F4"),
        ]:
            self._add_action(m, label, key,
                             lambda checked=False, p=page: self._nav(p))
        m.addSeparator()
        self._add_action(m, "Thème clair / sombre", "Ctrl+T",
                         lambda: self._js(
                             "document.getElementById('btn-theme-toggle')?.click()"))
        m.addSeparator()
        self._add_action(m, "Plein écran", "F11", self._toggle_fullscreen)
        self._add_action(m, "Zoom +",  "Ctrl++",
                         lambda: self.webview.setZoomFactor(
                             min(self.webview.zoomFactor() + 0.1, 3.0)))
        self._add_action(m, "Zoom −",  "Ctrl+-",
                         lambda: self.webview.setZoomFactor(
                             max(self.webview.zoomFactor() - 0.1, 0.5)))
        self._add_action(m, "Zoom par défaut", "Ctrl+0",
                         lambda: self.webview.setZoomFactor(1.0))
        m.addSeparator()
        self._add_action(m, "Recharger", "F5", self.webview.reload)

        # Exécuter
        m = mb.addMenu("Exécuter")
        self._add_action(m, "Lancer pipeline LOF*",  "Ctrl+R",
                         lambda: self._js(
                             "document.getElementById('btn-run-pipeline')?.click()"))
        self._add_action(m, "Lancer pipeline BiVAT", "Ctrl+B",
                         lambda: self._js(
                             "var s=document.getElementById('dd-modele');"
                             "if(s){s.value='bivat';"
                             "s.dispatchEvent(new Event('change',{bubbles:true}));}"
                             "setTimeout(()=>document.getElementById"
                             "('btn-run-pipeline')?.click(),300);"))
        m.addSeparator()
        self._add_action(m, "Arrêter l'analyse", "Ctrl+.",
                         lambda: self._js(
                             "document.getElementById('btn-stop-pipeline')?.click()"))

        # Aide
        m = mb.addMenu("Aide")
        self._add_action(m, "Guide d'utilisation", "F1",
                         lambda: show_aide(self))
        self._add_action(m, "Ouvrir les logs", None, self._open_logs)
        m.addSeparator()
        self._add_action(m, "À propos…", None, lambda: show_about(self))

    def _handle_download(self, download):
        reports_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "reports"))
        os.makedirs(reports_dir, exist_ok=True)
        fname = download.suggestedFileName() or "rapport_beac.html"
        download.setDownloadDirectory(reports_dir)
        download.setDownloadFileName(fname)
        download.accept()
        full_path = os.path.join(reports_dir, fname)
        QTimer.singleShot(800, lambda p=full_path: self._open_after_download(p))

    def _open_after_download(self, path: str):
        from PyQt6.QtWidgets import QMessageBox
        if os.path.exists(path):
            os.startfile(path)
        else:
            QMessageBox.information(self, "Rapport",
                f"Rapport enregistré dans :\n{os.path.dirname(path)}")

    def _add_action(self, menu, label, shortcut, slot):
        from PyQt6.QtGui import QAction, QKeySequence
        a = QAction(label, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        menu.addAction(a)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _js(self, code: str):
        self.webview.page().runJavaScript(code)

    def _nav(self, page_id: str):
        self._js(f"document.getElementById('nav-{page_id}')?.click()")

    def _toggle_max(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _poll_page(self):
        def _on_result(on_login):
            page = "login" if on_login else "app"
            if page == self._current_page:
                return
            self._current_page = page
            self.menuBar().setVisible(page != "login")
            self._custom_titlebar.setVisible(page == "login")
            sz = self._login_size if page == "login" else self._full_size
            self.resize(*sz)
            screen = QApplication.primaryScreen().geometry()
            self.move(
                (screen.width()  - sz[0]) // 2,
                (screen.height() - sz[1]) // 2,
            )
            # Une fois dans l'app, la transition suivante (déconnexion) est un
            # événement utilisateur rare — inutile de continuer à interroger le
            # DOM 800ms/tick pendant toute la session ; on ralentit le polling.
            # Il redevient réactif dès le retour à l'écran de login.
            self._page_timer.setInterval(800 if page == "login" else 4000)

        self.webview.page().runJavaScript(
            "!!document.getElementById('login-screen')",
            _on_result,
        )

    # ── Actions ───────────────────────────────────────────────────────────────

    def _import_xlsx(self):
        data_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data"))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Importer fichier(s) XLSX", data_dir,
            "Fichiers Excel (*.xlsx *.xls)",
        )
        if not paths:
            return
        os.makedirs(data_dir, exist_ok=True)
        for p in paths:
            dst = os.path.join(data_dir, os.path.basename(p))
            if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(dst)):
                shutil.copy2(p, dst)
        self._nav("donnees")

    def _export_figures(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Dossier d'export",
            os.path.normpath(os.path.join(os.path.dirname(__file__), "..")),
        )
        if folder:
            self._js(
                f"window._beac_export_dir='{folder.replace(chr(92),'/')}'; "
                "document.getElementById('btn-export-figures')?.click();"
            )

    def _open_logs(self):
        log_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "logs", "beac.log"))
        if os.path.exists(log_path):
            os.startfile(log_path)
        else:
            QMessageBox.information(
                self, "Logs", "Aucun fichier de log lancez l'application.")
