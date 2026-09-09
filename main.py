"""BEAC Anomaly Detector point d'entrée desktop (PyQt6 + Dash).

Usage:
    python main.py          # production
    python app.py           # développement (hot-reload Dash)
"""
import os
import sys
import threading
import time

os.environ.setdefault("BEAC_DEV", "0")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon, QColor
from PyQt6.QtCore import Qt

from qt.splash import make_splash
from qt.window import BEACWindow

HOST = "127.0.0.1"
PORT = 8050
URL  = f"http://{HOST}:{PORT}"
ICON = os.path.join(os.path.dirname(__file__), "assets", "logo_beac.jfif")


def _start_dash_server():
    from app import server
    from waitress import serve
    serve(server, host=HOST, port=PORT, threads=4)


def _wait_for_server(timeout: int = 20) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(URL, timeout=1)
            return True
        except Exception:
            time.sleep(0.25)
    return False


def main():
    dash_thread = threading.Thread(
        target=_start_dash_server, daemon=True, name="dash-server")
    dash_thread.start()

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("BEAC Anomaly Detector")
    qt_app.setOrganizationName("BEAC-CEMAC")
    if os.path.exists(ICON):
        qt_app.setWindowIcon(QIcon(ICON))

    splash = make_splash()
    splash.show()
    qt_app.processEvents()

    if not _wait_for_server(timeout=20):
        splash.showMessage(
            "Erreur : impossible de démarrer le serveur Dash.",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            QColor("#E05252"),
        )
        qt_app.processEvents()
        time.sleep(3)
        sys.exit(1)

    window = BEACWindow(url=URL, dev=False)
    splash.finish(window)
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
