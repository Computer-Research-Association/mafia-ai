from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal
from .tabs.safeComboBox import SafeComboBox


class AgentConfigWidget(QGroupBox):
    """각 플레이어(0~7)를 개별 설정하는 위젯"""

    typeChanged = pyqtSignal()

    def __init__(self, player_id):
        super().__init__(f"Player {player_id}")
        self.player_id = player_id
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 1. 상단 공통 설정 (Type 및 Role)
        top_layout = QHBoxLayout()

        # [Type 설정]
        top_layout.addWidget(QLabel("Type:"))
        self.type_combo = SafeComboBox()
        self.type_combo.addItems(["RL", "LLM", "RBA"])
        self.type_combo.setSizePolicy(
            self.type_combo.sizePolicy().horizontalPolicy(),
            self.type_combo.sizePolicy().verticalPolicy(),
        )
        top_layout.addWidget(self.type_combo, stretch=1)

        # [Role 설정] - 공통 영역으로 이동됨
        top_layout.addWidget(QLabel("Role:"))
        self.role_combo = SafeComboBox()
        # Random을 기본값으로 사용하기 위해 맨 앞에 추가
        self.role_combo.addItems(["Random", "Citizen", "Police", "Doctor", "Mafia"])
        top_layout.addWidget(self.role_combo, stretch=1)

        self.layout.addLayout(top_layout)

        # 2. RL 전용 설정 영역 (RL 선택 시에만 보임)
        self.rl_config_area = QWidget()
        rl_layout = QVBoxLayout()
        self.rl_config_area.setLayout(rl_layout)
        rl_layout.setContentsMargins(0, 0, 0, 0)  # 내부 여백 제거

        # [모델 불러오기 설정]
        model_load_layout = QHBoxLayout()
        rl_layout.addWidget(QLabel("Load Model:"))

        self.load_model_path_input = QLineEdit()
        self.load_model_path_input.setPlaceholderText("선택 안 함 (처음부터 학습)")
        self.load_model_path_input.setReadOnly(True)
        model_load_layout.addWidget(self.load_model_path_input)

        self.btn_select_model = QPushButton("📂")
        self.btn_select_model.setFixedWidth(30)
        self.btn_select_model.clicked.connect(self._select_model_file)
        model_load_layout.addWidget(self.btn_select_model)

        rl_layout.addLayout(model_load_layout)

        # 초기화 버튼
        self.btn_clear_model = QPushButton("❌")
        self.btn_clear_model.setFixedWidth(30)
        self.btn_clear_model.setToolTip("모델 선택 해제")
        self.btn_clear_model.clicked.connect(self._clear_model_file)
        model_load_layout.addWidget(self.btn_clear_model)

        # 모델 선택시 숨겨지는 컨테이너
        self.param_container = QWidget()
        self.param_layout = QVBoxLayout(self.param_container)
        self.param_layout.setContentsMargins(0, 0, 0, 0)  # 여백 정리

        # [알고리즘]
        self.param_layout.addWidget(QLabel("Algorithm:"))
        self.algo_combo = SafeComboBox()
        self.algo_combo.addItems(["PPO", "REINFORCE"])
        self.param_layout.addWidget(self.algo_combo)

        # [백본]
        self.param_layout.addWidget(QLabel("Backbone:"))
        self.backbone_combo = SafeComboBox()
        self.backbone_combo.addItems(["MLP", "LSTM", "GRU"])
        self.param_layout.addWidget(self.backbone_combo)

        # 파라미터 컨테이너를 RL 영역에 추가
        rl_layout.addWidget(self.param_container)

        # [중요 해결] rl_config_area를 메인 레이아웃에 추가해야 새 창이 안 뜹니다!
        self.layout.addWidget(self.rl_config_area)

        self.layout.addStretch()

        # 초기 상태 설정
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self._toggle_rl_area(self.type_combo.currentText())

    def _select_model_file(self):
        """모델 파일(.pt) 선택 시 파라미터 숨김 처리"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "학습된 모델 파일 선택",
            "./models",
            "Model Files (*.pt);;All Files (*)",
        )
        if file_path:
            self.load_model_path_input.setText(file_path)
            try:
                import torch

                checkpoint = torch.load(file_path, map_location="cpu")
                saved_algo = checkpoint.get("algorithm", "PPO")
                saved_backbone = checkpoint.get("backbone", "MLP")
                self.algo_combo.setCurrentText(saved_algo.upper())
                self.backbone_combo.setCurrentText(saved_backbone.upper())

            except Exception as e:
                print(f"[GUI] Warning: Failed to read metadata from model: {e}")

            # 파일이 선택되면 파라미터 설정창 숨기기
            self.param_container.setVisible(False)

    def _clear_model_file(self):
        """모델 선택 해제 시 파라미터 다시 보이기"""
        self.load_model_path_input.clear()
        # [핵심 기능] 파일이 해제되면 파라미터 설정창 보이기
        self.param_container.setVisible(True)

    def _on_type_changed(self, text):
        self._toggle_rl_area(text)
        self.typeChanged.emit()

    def _clear_model_file(self):
        self.load_model_path_input.clear()
        self.param_container.setVisible(True)

    def _on_type_changed(self, text):
        self._toggle_rl_area(text)
        self.typeChanged.emit()

    def _toggle_rl_area(self, agent_type):
        """에이전트 타입에 따라 RL 설정 영역 표시/숨김"""
        self.rl_config_area.setVisible(agent_type == "RL")

    def get_config(self):
        """현재 설정된 에이전트 정보를 딕셔너리로 반환"""
        config = {"type": self.type_combo.currentText().lower()}
        config["role"] = self.role_combo.currentText().lower()

        if config["type"] == "rl":
            # 모델 경로 가져오기
            path_text = self.load_model_path_input.text().strip()
            config["load_model_path"] = path_text if path_text else None

            if config["load_model_path"]:
                # 모델을 로드하는 경우:
                config["algo"] = self.algo_combo.currentText().lower()
                config["backbone"] = self.backbone_combo.currentText().lower()
            else:
                # 모델 파일이 없는 경우
                config["algo"] = self.algo_combo.currentText().lower()
                config["backbone"] = self.backbone_combo.currentText().lower()

        return config

    def set_config(
        self,
        agent_type="LLM",
        role="Random",  # [추가] Role 설정 인자
        algo="PPO",
        backbone="MLP",  # Change default to MLP
        load_model_path=None,  # [추가] 모델 경로 인자
    ):
        """외부에서 설정을 일괄 적용할 때 사용"""
        self.type_combo.setCurrentText(agent_type.upper())

        # [추가] Role 설정 반영
        role_text = role.capitalize()
        if self.role_combo.findText(role_text) >= 0:
            self.role_combo.setCurrentText(role_text)
        else:
            self.role_combo.setCurrentText("Random")

        if agent_type.upper() == "RL":
            self.algo_combo.setCurrentText(algo.upper())
            self.backbone_combo.setCurrentText(backbone.upper())

            if load_model_path:
                self.load_model_path_input.setText(load_model_path)
