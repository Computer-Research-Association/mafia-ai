import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtGui import QFont

from .tabs.log_viewer import LogViewerTab


class MafiaLogViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mafia AI 게임 로그 뷰어")
        self.resize(1100, 750)
        self.setFont(QFont("Malgun Gothic", 10))  # 폰트 설정
        # 중앙 위젯, 레이아웃 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        # 탭 추가
        self.log_viewer_tab = LogViewerTab(self)
        layout.addWidget(self.log_viewer_tab)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MafiaLogViewerWindow()
    window.show()
    sys.exit(app.exec())
