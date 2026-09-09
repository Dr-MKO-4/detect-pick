"""BEACMenuBar menubar draggable avec style Qt."""
from PyQt6.QtWidgets import QMenuBar
from PyQt6.QtCore import Qt, QPoint

MENUBAR_STYLE = """
QMenuBar {
    background: #f5f3ee;
    color: #1a1a1a;
    font-family: "Segoe UI", "Roboto", sans-serif;
    font-size: 12px;
    border-bottom: 1px solid #ccc8bc;
    padding: 1px 0;
}
QMenuBar::item { padding: 4px 10px; background: transparent; border-radius: 3px; }
QMenuBar::item:selected { background: rgba(139,90,0,.12); color: #7a4f00; }
QMenu {
    background: #ffffff; color: #1a1a1a;
    border: 1px solid #ccc8bc; border-radius: 5px; padding: 4px 0;
    font-size: 12px; font-family: "Segoe UI", "Roboto", sans-serif;
}
QMenu::item { padding: 6px 28px 6px 16px; border-radius: 3px; margin: 1px 4px; }
QMenu::item:selected { background: rgba(139,90,0,.10); color: #7a4f00; }
QMenu::item:disabled { color: #aaa; }
QMenu::separator { height: 1px; background: #e5e1d8; margin: 3px 8px; }
"""


class _DraggableMenuBar:
    _drag_pos: QPoint | None = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (event.globalPosition().toPoint()
                              - self.window().frameGeometry().topLeft())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            w = self.window()
            w.showNormal() if w.isMaximized() else w.showMaximized()
        super().mouseDoubleClickEvent(event)


class BEACMenuBar(_DraggableMenuBar, QMenuBar):
    pass
