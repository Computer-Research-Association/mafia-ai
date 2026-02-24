# 🧩 Apps (응용 어플리케이션 및 유틸리티)

<div align="center">
  시뮬레이터 코어 시스템을 바탕으로 구축된 부가적인 응용 프로그램이나 <br/>
  인간 개입(Human Interaction) 연구를 지원하는 모듈들이 모여있는 디렉토리입니다.
</div>

---

## 📂 파일 구성 (File Structure)

### 🧑‍💻 `human_interaction/`

에이전트 단독 시뮬레이션이 아닌, 인간 플레이어가 직접 관여하거나 프롬프트를 조정하며 테스트해보기 위한 도구들입니다.

```text
apps/human_interaction/
├── main.py        # 인간 상호작용 세션 실행 메인 스크립트
├── game.py        # 상호작용 가능한 형태의 게임 래퍼 로직
├── agent.py       # 인간의 입력을 받아 동작하는 휴먼 에이전트 클래스
└── state.py       # 상호작용 중 발생하는 특수 상태 관리 모듈
```

---

⬅️ **[메인으로 돌아가기](../README.md)**
