# GUI System (인터페이스 및 시각화)

<div align="center">
  사용자가 마피아 게임 시뮬레이션을 쉽게 설정하고 실행하며,<br/>
  결과를 시각적으로 분석할 수 있는 <b>PyQt6 기반 모듈</b>입니다.
</div>

---

## 파일 구성 (File Structure)

### 1. 런처 (Launcher)

* **`launcher.py`**: 프로그램 진입점. 학습/테스트 모드 선택, 에피소드 수 지정, 경로 설정 등을 종합적으로 담당합니다.
* **`agentConfig.py`**: 참가하는 8명의 플레이어 각각에 대해 에이전트 종류(`RL`, `LLM`, `RBA`), 역할(`Role`), 모델 백본 등을 개별적으로 구성하는 위젯 모듈입니다.
* **`styles.qss`**: GUI 전체의 일관된 디자인 테마(Dark Mode 중심)를 호스팅하는 스타일시트입니다.

### 2. 로그 뷰어 (Log Viewer)

```text
gui/
├── gui_viewer.py            # 로그 뷰어의 메인 윈도우 컨트롤러
└── tabs/
    └── log_viewer/
        ├── logViewer.py     # 좌측(탐색기) + 우측(뷰어) 통합 관리 및 TensorBoard 제어
        ├── logLeft.py       # `logs/` 폴더 내 JSONL 파일 및 텐서보드를 읽어오는 트리 뷰
        ├── logRight.py      # 선택된 게임 로그를 날짜(Day)와 페이즈(Phase)별로 가독성 있게 렌더링
        └── logEvent.py      # 파싱된 로그 데이터를 구조화하는 모델 데이터
```

---

## ▶️ 실행 방법 (Usage)

루트 디렉토리의 `main.py`를 실행하면 시뮬레이션 설정과 시각화를 위한 런처(GUI)가 즉시 시작됩니다.

```bash
# 프로젝트 루트 디렉토리에서 실행
python main.py
```

---

⬅️ **[메인으로 돌아가기](../README.md)**