from pathlib import Path
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QIcon


class StyleLoader:
    @staticmethod
    def load_stylesheet(widget: QWidget, filename: str = "styles.qss"):
        """assets 폴더에서 qss 파일을 찾아 위젯에 적용"""
        current_dir = Path(__file__).parent
        qss_path = current_dir.parent / "assets" / filename

        if qss_path.exists():
            with open(qss_path, "r", encoding="utf-8") as f:
                widget.setStyleSheet(f.read())
        else:
            print(f"Warning: Stylesheet not found at {qss_path}")

    @staticmethod
    def get_icon_path(filename: str = "icon.jpg") -> str:
        """assets 폴더에서 아이콘 경로 반환"""
        current_dir = Path(__file__).parent
        icon_path = current_dir.parent / "assets" / filename
        return str(icon_path) if icon_path.exists() else ""
