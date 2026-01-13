# GUI System

사용자가 마피아 게임 시뮬레이션을 쉽게 설정하고 실행하며, 결과를 시각적으로 분석할 수 있는 PyQt6 기반 인터페이스입니다.

## 파일 구성

### 1. 런처 (Launcher)
- **`launcher.py`**: 프로그램 진입점. 학습/테스트 모드 선택, 에피소드 수 설정, 경로 설정 등을 담당.
- **`agentConfig.py`**: 8명의 플레이어 각각에 대해 에이전트 타입(RL, LLM, RBA), 역할(Role), 모델 백본 등을 개별 설정하는 위젯.
- **`styles.qss`**: GUI 전체의 디자인 테마(Dark Mode) 정의.

### 2. 로그 뷰어 (Log Viewer)
- **`gui_viewer.py`**: 로그 뷰어 윈도우 메인 클래스.
- **`tabs/log_viewer/`**:
  - **`logViewer.py`**: 좌측 탐색기와 우측 뷰어를 통합 관리하며 TensorBoard 프로세스를 제어.
  - **`logLeft.py`**: `logs/` 디렉토리의 JSONL 로그 파일 및 텐서보드 로그를 탐색하는 트리 뷰.
  - **`logRight.py`**: 선택된 게임 로그를 파싱하여 날짜(Day)와 페이즈(Phase)별로 가독성 있게 출력. 필터링 기능 제공.
  - **`logEvent.py`**: 로그 파싱을 위한 데이터 모델.

### 실행 방법
루트 디렉토리의 `main.py`를 실행하면 런처가 시작됩니다.