from PyQt6.QtWidgets import QCheckBox


class ILButton(QCheckBox):
    def __init__(self, parent=None):
        super().__init__("IL 모드", parent)
        self.setChecked(False)

    def is_enabled(self) -> bool:
        """IL 모드가 켜져 있는지 확인"""
        return self.isChecked()
