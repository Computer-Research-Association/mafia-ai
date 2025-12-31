from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QMessageBox,
    QLineEdit,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal, Qt
from argparse import Namespace
from pathlib import Path


class AgentConfigWidget(QGroupBox):
    """각 플레이어(0~7)를 개별 설정하는 위젯"""
    
    def __init__(self, player_id):
        super().__init__(f"Player {player_id}")
        self.player_id = player_id
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 1. 에이전트 메인 타입 (LLM vs RL)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["LLM", "RL"])
        self.layout.addWidget(QLabel("Type:"))
        self.layout.addWidget(self.type_combo)
        
        # 2. RL 전용 설정 영역 (RL 선택 시만 노출/활성화)
        self.rl_config_area = QWidget()
        rl_layout = QVBoxLayout()
        self.rl_config_area.setLayout(rl_layout)
        
        # 알고리즘 선택
        rl_layout.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["PPO", "REINFORCE"])
        rl_layout.addWidget(self.algo_combo)
        
        # 백본 선택
        rl_layout.addWidget(QLabel("Backbone:"))
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(["MLP", "LSTM", "GRU"])
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
        self.type_combo.currentTextChanged.connect(self._toggle_rl_area)
        self._toggle_rl_area(self.type_combo.currentText())
    
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
    
    def set_config(self, agent_type="LLM", algo="PPO", backbone="MLP", hidden_dim=128, num_layers=2):
        """외부에서 설정을 일괄 적용할 때 사용"""
        self.type_combo.setCurrentText(agent_type.upper())
        if agent_type.upper() == "RL":
            self.algo_combo.setCurrentText(algo.upper())
            self.backbone_combo.setCurrentText(backbone.upper())
            self.hidden_dim_spin.setValue(hidden_dim)
            self.num_layers_spin.setValue(num_layers)


class Launcher(QWidget):
    start_simulation_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mafia AI Simulation")
        self.resize(400, 450)

        # 8개의 개별 에이전트 설정 위젯을 저장
        self.agent_config_widgets = []

        self._init_ui()

    def _init_ui(self):
        # === [메인 레이아웃] ===
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        # =================================================
        # [왼쪽 패널]
        # =================================================
        self.left_widget = QWidget()
        layout = QVBoxLayout()
        self.left_widget.setLayout(layout)

        title = QLabel("마피아 AI 시물레이터")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        # 1. 실행 모드
        mode_group = QGroupBox("실행 모드")
        mode_layout = QHBoxLayout()
        self.radio_train = QRadioButton("학습 (Train)")
        self.radio_test = QRadioButton("평가 (Test)")
        self.radio_test.setChecked(True)

        btn_group = QButtonGroup(self)
        btn_group.addButton(self.radio_train)
        btn_group.addButton(self.radio_test)

        mode_layout.addWidget(self.radio_train)
        mode_layout.addWidget(self.radio_test)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 2. 에피소드 수
        ep_group = QGroupBox("진행 에피소드 수")
        ep_layout = QVBoxLayout()
        self.ep_spin = QSpinBox()
        self.ep_spin.setRange(1, 10000)
        self.ep_spin.setValue(1)
        ep_layout.addWidget(self.ep_spin)
        ep_group.setLayout(ep_layout)
        layout.addWidget(ep_group)
        
        # 3. 빠른 설정 (일괄 적용)
        quick_group = QGroupBox("빠른 설정")
        quick_layout = QVBoxLayout()
        
        quick_desc = QLabel("모든 플레이어에게 동일한 설정 일괄 적용")
        quick_desc.setStyleSheet("color: gray; font-size: 11px;")
        quick_layout.addWidget(quick_desc)
        
        quick_controls = QHBoxLayout()
        
        self.quick_type_combo = QComboBox()
        self.quick_type_combo.addItems(["LLM", "RL"])
        quick_controls.addWidget(QLabel("Type:"))
        quick_controls.addWidget(self.quick_type_combo)
        
        btn_apply_all = QPushButton("모두 적용")
        btn_apply_all.clicked.connect(self.apply_to_all_agents)
        quick_controls.addWidget(btn_apply_all)
        
        quick_layout.addLayout(quick_controls)
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        # 4. 경로 관리
        path_group = QGroupBox("경로 관리")
        path_layout = QGridLayout()
        
        # 모델 저장 경로
        path_layout.addWidget(QLabel("모델 저장:"), 0, 0)
        self.model_path_input = QLineEdit()
        self.model_path_input.setText("./models")
        self.model_path_input.setReadOnly(True)
        path_layout.addWidget(self.model_path_input, 0, 1)
        
        btn_model_path = QPushButton("📁")
        btn_model_path.setFixedSize(30, 30)
        btn_model_path.clicked.connect(self.select_model_path)
        path_layout.addWidget(btn_model_path, 0, 2)
        
        # 로그 출력 경로
        path_layout.addWidget(QLabel("로그 출력:"), 1, 0)
        self.log_path_input = QLineEdit()
        self.log_path_input.setText("./logs")
        self.log_path_input.setReadOnly(True)
        path_layout.addWidget(self.log_path_input, 1, 1)
        
        btn_log_path = QPushButton("📁")
        btn_log_path.setFixedSize(30, 30)
        btn_log_path.clicked.connect(self.select_log_path)
        path_layout.addWidget(btn_log_path, 1, 2)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        layout.addStretch()
        
        # 에이전트 설정 버튼
        self.btn_expand = QPushButton("⚙️ 개별 에이전트 상세 설정")
        self.btn_expand.setCheckable(True)
        self.btn_expand.setToolTip("8명의 에이전트를 개별적으로 설정합니다")
        self.btn_expand.clicked.connect(self.toggle_right_panel)
        layout.addWidget(self.btn_expand)

        # 시작 버튼
        self.btn_start = QPushButton("시뮬레이션 시작")
        self.btn_start.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-size: 16px; 
                padding: 12px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #45a049; }
        """
        )
        self.btn_start.clicked.connect(self.on_click_start)
        layout.addWidget(self.btn_start)

        # =================================================
        # [오른쪽 패널] - 8개의 독립적인 에이전트 설정
        # =================================================
        self.right_panel = QGroupBox("개별 에이전트 설정 (8명)")
        self.right_panel.setVisible(False)

        right_layout = QGridLayout()
        self.right_panel.setLayout(right_layout)

        # 8개의 AgentConfigWidget 생성
        for i in range(8):
            agent_widget = AgentConfigWidget(i)
            self.agent_config_widgets.append(agent_widget)
            
            row = i // 2
            col = i % 2
            right_layout.addWidget(agent_widget, row, col)

        self.main_layout.addWidget(self.left_widget)
        self.main_layout.addWidget(self.right_panel)

    def toggle_right_panel(self):
        """설정 버튼 클릭 시 패널 열기/닫기"""
        if self.btn_expand.isChecked():
            self.right_panel.setVisible(True)
            self.resize(1100, 700)
        else:
            self.right_panel.setVisible(False)
            self.resize(400, 550)
            self.adjustSize()
    
    def apply_to_all_agents(self):
        """빠른 설정을 모든 에이전트에 일괄 적용"""
        agent_type = self.quick_type_combo.currentText()
        
        for widget in self.agent_config_widgets:
            widget.set_config(agent_type=agent_type)
        
        QMessageBox.information(
            self,
            "설정 적용 완료",
            f"모든 플레이어를 {agent_type}로 설정했습니다."
        )
    
    def select_model_path(self):
        """모델 저장 경로 선택"""
        path = QFileDialog.getExistingDirectory(self, "모델 저장 경로 선택", self.model_path_input.text())
        if path:
            self.model_path_input.setText(path)
    
    def select_log_path(self):
        """로그 출력 경로 선택"""
        path = QFileDialog.getExistingDirectory(self, "로그 출력 경로 선택", self.log_path_input.text())
        if path:
            self.log_path_input.setText(path)

    def on_click_start(self):
        """시뮬레이션 시작 버튼 클릭 - 개별 에이전트 설정 수집"""
        
        # 8개 에이전트의 개별 설정 수집
        player_configs = [widget.get_config() for widget in self.agent_config_widgets]
        
        mode = "train" if self.radio_train.isChecked() else "test"
        
        # 경로 설정
        paths = {
            "model_dir": Path(self.model_path_input.text()),
            "log_dir": Path(self.log_path_input.text()),
        }

        args = Namespace(
            player_configs=player_configs,  # 새로운 구조!
            mode=mode,
            episodes=self.ep_spin.value(),
            gui=True,
            paths=paths,
        )

        self.start_simulation_signal.emit(args)
