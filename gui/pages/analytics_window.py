import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget
from PyQt6.QtGui import QFont
from PyQt6.QtGui import QIcon

from .analytics_view import AnalyticsView
from pathlib import Path
from gui.utils.style_loader import StyleLoader


class AnalyticsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mafia AI 게임 로그 뷰어")
        # Set window icon
        icon_path = StyleLoader.get_icon_path("icon.jpg")
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        self.resize(1100, 750)
        # stylesheet
        StyleLoader.load_stylesheet(self, "styles.qss")
        # 중앙 위젯, 레이아웃 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.analytics_view = AnalyticsView(self)
        self.tab_widget.addTab(self.analytics_view, "로그 뷰어")

    def show_live(self, log_path):
        self.tab_widget.setCurrentWidget(self.analytics_view)
        self.analytics_view.select_live(log_path)

    def closeEvent(self, event):
        if hasattr(self, "analytics_view") and self.analytics_view:
            self.analytics_view.shutdown_tensorboard()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalyticsWindow()
    window.show()
    sys.exit(app.exec())
