from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QMessageBox,
)
from PyQt6.QtCore import pyqtSignal, Qt
from argparse import Namespace


class Launcher(QWidget):
    start_simulation_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mafia AI Simulation")
        self.setGeometry(100, 100, 400, 400)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("마피아 AI 시물레이터")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        # 1. 에이전트 선택
        agent_group = QGroupBox("플레이어 에이전트")
        agent_layout = QVBoxLayout()
        self.agent_combo = QComboBox()
        self.agent_combo.addItems(["llm", "ppo", "reinforce"])
        agent_layout.addWidget(self.agent_combo)
        agent_group.setLayout(agent_layout)
        layout.addWidget(agent_group)

        # 2. 실행 모드 선택
        mode_group = QGroupBox("실행 모드")
        mode_layout = QHBoxLayout()
        self.radio_train = QRadioButton("학습 (Train)")
        self.radio_test = QRadioButton("실습/테스트 (Test)")
        self.radio_test.setChecked(True)

        btn_group = QButtonGroup(self)  # 라디오 버튼 그룹핑
        btn_group.addButton(self.radio_train)
        btn_group.addButton(self.radio_test)

        mode_layout.addWidget(self.radio_train)
        mode_layout.addWidget(self.radio_test)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        ep_group = QGroupBox("진행 에피소드 수")
        ep_layout = QVBoxLayout()
        self.ep_spin = QSpinBox()
        self.ep_spin.setRange(1, 10000)
        self.ep_spin.setValue(10)
        ep_layout.addWidget(self.ep_spin)
        ep_group.setLayout(ep_layout)
        layout.addWidget(ep_group)

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

        self.setLayout(layout)

    def on_click_start(self):
        agent = self.agent_combo.currentText()
        mode = "train" if self.radio_train.isChecked() else "test"

        if agent == "llm" and mode == "train":
            QMessageBox.warning(
                self,
                "주의",
                "LLM 에이전트는 학습 모드를 지원하지 않습니다.\nTest 모드로 진행해주세요.",
            )
            return

        args = Namespace(
            agent=agent, mode=mode, episodes=self.ep_spin.value(), gui=True
        )

        self.start_simulation_signal.emit(args)

    def start(self):
        self.show()


# if __name__ == "__main__":
#     import sys
#     from PyQt6.QtWidgets import QApplication

#     # 1. 어플리케이션 객체 생성 (필수)
#     app = QApplication(sys.argv)

#     # 2. 런처 생성 및 실행
#     launcher = Launcher()
#     launcher.start()  # self.show() 호출

#     # 3. 이벤트 루프 실행 (창이 꺼지지 않게 유지)
#     sys.exit(app.exec())
