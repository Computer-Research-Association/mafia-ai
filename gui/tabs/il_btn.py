from PyQt6.QtWidgets import QCheckBox


class ILButton(QCheckBox):
    def __init__(self, parent=None):
        super().__init__("Imitation Learning (IL) 모드 켜기", parent)
        self.setChecked(False)  # 기본값: OFF
        self.setToolTip("활성화 시 기존 로그 데이터를 활용하여 학습을 가속화합니다.")
        # 스타일링이 필요하다면 여기에 setStyleSheet 등을 추가할 수 있습니다.

    def is_enabled(self) -> bool:
        """IL 모드가 켜져 있는지 확인"""
        return self.isChecked()
