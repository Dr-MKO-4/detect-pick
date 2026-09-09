"""Splash screen BEAC au démarrage."""
import os

from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtGui import (
    QPixmap, QColor, QPainter, QBrush, QPen,
    QLinearGradient, QFont,
)
from PyQt6.QtCore import Qt, QRect

_ICON = os.path.join(os.path.dirname(__file__), "..", "assets", "logo_beac.jfif")


def make_splash() -> QSplashScreen:
    W, H = 520, 300
    pix = QPixmap(W, H)

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    grad = QLinearGradient(0, 0, 0, H)
    grad.setColorAt(0, QColor("#111010"))
    grad.setColorAt(1, QColor("#0a0907"))
    painter.fillRect(0, 0, W, H, QBrush(grad))

    painter.setPen(QPen(QColor("#D4A020"), 3))
    painter.drawLine(0, 0, W, 0)

    logo_size = 64
    logo_x = (W - logo_size) // 2
    logo_y = 44
    if os.path.exists(_ICON):
        logo_pix = QPixmap(_ICON).scaled(
            logo_size, logo_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(
            logo_x + (logo_size - logo_pix.width()) // 2,
            logo_y + (logo_size - logo_pix.height()) // 2,
            logo_pix,
        )
    else:
        painter.setBrush(QBrush(QColor("#D4A020")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(logo_x, logo_y, logo_size, logo_size)

    title_font = QFont("Segoe UI", 16, QFont.Weight.DemiBold)
    painter.setFont(title_font)
    painter.setPen(QColor("#e8e6e2"))
    painter.drawText(QRect(0, logo_y + logo_size + 16, W, 32),
                     Qt.AlignmentFlag.AlignHCenter, "BEAC · CEMAC")

    sub_font = QFont("Segoe UI", 10, QFont.Weight.Normal)
    painter.setFont(sub_font)
    painter.setPen(QColor("#9e8c6a"))
    painter.drawText(QRect(0, logo_y + logo_size + 50, W, 22),
                     Qt.AlignmentFlag.AlignHCenter, "Détection d'anomalies monétaires")

    painter.setPen(QPen(QColor("#2a2826"), 1))
    painter.drawLine(60, H - 68, W - 60, H - 68)

    bar_x, bar_y, bar_w, bar_h = 60, H - 52, W - 120, 3
    painter.setBrush(QBrush(QColor("#2a2826")))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRect(bar_x, bar_y, bar_w, bar_h)
    painter.setBrush(QBrush(QColor("#D4A020")))
    painter.drawRect(bar_x, bar_y, int(bar_w * 0.40), bar_h)

    ver_font = QFont("Segoe UI", 8, QFont.Weight.Normal)
    painter.setFont(ver_font)
    painter.setPen(QColor("#5a5650"))
    painter.drawText(QRect(0, H - 22, W, 18),
                     Qt.AlignmentFlag.AlignHCenter,
                     "v1.0  ·  Direction de la Recherche et des Statistiques")

    painter.end()

    splash = QSplashScreen(pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.showMessage(
        "Démarrage en cours…",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        QColor("#D4A020"),
    )
    return splash
