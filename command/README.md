# Command (터미널 스크립트 도구)

<div align="center">
  GUI를 통하지 않고 CUI 환경이나 백그라운드 서버에서 직접 실행하기 위한 <br/>
  학습, 테스트 파이프라인 및 최적화(Optuna) 스크립트들의 모음입니다.
</div>

---

## 주요 파일 (Key Scripts)

**`train_graph.py` & `test_graph.py`**
  * 모델을 터미널에서 직접 훈련(`train`)하거나 성능을 테스트(`test`)하기 위해 설계된 단독 실행 스크립트들입니다.
  * 하이퍼파라미터 및 에이전트 구성을 코드로 직접 제어하며, CI/CD 환경이나 대규모 서버 배치 학습에서 주로 활용됩니다.

**`optimize.py`**
  * `Optuna` 프레임워크를 기반으로 최적의 하이퍼파라미터(Learning Rate, Batch Size 등) 서치를 자동화하는 모듈입니다.

**`run_rq1.py` & `run_rq1_test.py`**
  * 연구 퀘스천 1번(RQ1: 특정 학습 방법론과 수렴/성능의 관계 등)을 검증하기 위해 미리 구성된 실험용 파이프라인 스크립트입니다.

**`run_rq1_pipeline.bat`**
  * 윈도우(Windows) 환경에서 위의 실험들을 연속적으로 처리하기 위한 일괄 실행 배치 파일입니다.

---

⬅️ **[메인으로 돌아가기](../README.md)**
