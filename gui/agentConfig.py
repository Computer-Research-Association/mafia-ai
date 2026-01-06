from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QGroupBox,
)
from PyQt6.QtCore import pyqtSignal


class AgentConfigWidget(QGroupBox):
    """각 플레이어(0~7)를 개별 설정하는 위젯"""

    typeChanged = pyqtSignal()

    def __init__(self, player_id):
        super().__init__(f"Player {player_id}")
        self.player_id = player_id
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Type:"))

        # 1. 에이전트 메인 타입 (LLM vs RL)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["LLM", "RL"])
        self.type_combo.setSizePolicy(
            self.type_combo.sizePolicy().horizontalPolicy(),
            self.type_combo.sizePolicy().verticalPolicy(),
        )
        top_layout.addWidget(self.type_combo, stretch=1)

        self.layout.addLayout(top_layout)

        # 2. RL 전용 설정 영역 (RL 선택 시만 노출/활성화)
        self.rl_config_area = QWidget()
        rl_layout = QVBoxLayout()
        self.rl_config_area.setLayout(rl_layout)
        rl_layout.setContentsMargins(0, 0, 0, 0)  # 내부 여백 제거

        # 알고리즘 선택
        rl_layout.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["PPO", "REINFORCE"])
        rl_layout.addWidget(self.algo_combo)

        # 백본 선택
        rl_layout.addWidget(QLabel("Backbone:"))
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(["LSTM", "GRU"])
        rl_layout.addWidget(self.backbone_combo)

        # 은닉층 차원
        rl_layout.addWidget(QLabel("Hidden Dim:"))
        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(32, 512)
        self.hidden_dim_spin.setValue(128)
        rl_layout.addWidget(self.hidden_dim_spin)

        # RNN 레이어 수 (LSTM/GRU용)
        rl_layout.addWidget(QLabel("RNN Layers:"))
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 4)
        self.num_layers_spin.setValue(2)
        rl_layout.addWidget(self.num_layers_spin)

        self.layout.addWidget(self.rl_config_area)

        # 타입 변경 시 RL 설정 영역 토글
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self._toggle_rl_area(self.type_combo.currentText())

        self.layout.addStretch()

    def _on_type_changed(self, text):
        self._toggle_rl_area(text)
        self.typeChanged.emit()

    def _toggle_rl_area(self, agent_type):
        """에이전트 타입에 따라 RL 설정 영역 표시/숨김"""
        self.rl_config_area.setVisible(agent_type == "RL")

    def get_config(self):
        """현재 설정된 에이전트 정보를 딕셔너리로 반환"""
        config = {"type": self.type_combo.currentText().lower()}
        if config["type"] == "rl":
            config["algo"] = self.algo_combo.currentText().lower()
            config["backbone"] = self.backbone_combo.currentText().lower()
            config["hidden_dim"] = self.hidden_dim_spin.value()
            config["num_layers"] = self.num_layers_spin.value()
        return config

    def set_config(
        self,
        agent_type="LLM",
        algo="PPO",
        backbone="LSTM",
        hidden_dim=128,
        num_layers=2,
    ):
        """외부에서 설정을 일괄 적용할 때 사용"""
        self.type_combo.setCurrentText(agent_type.upper())
        if agent_type.upper() == "RL":
            self.algo_combo.setCurrentText(algo.upper())
            self.hidden_dim_spin.setValue(hidden_dim)
            self.num_layers_spin.setValue(num_layers)
