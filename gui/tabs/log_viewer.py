"""
로그 뷰어 탭: 게임 이벤트를 자연어로 표시
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path
import json
from typing import List, Optional
from collections import defaultdict

from state import GameEvent
from config import Role, Phase, EventType
from core.logger import LogManager


class LogViewerTab:
    """게임 로그를 자연어로 표시하는 탭"""

    def __init__(self, notebook, log_file_path: str):
        """
        Args:
            notebook: 부모 노트북 위젯
            log_file_path: 로그 디렉토리 경로 (사용 안 함 - 직접 선택)
        """
        self.frame = ttk.Frame(notebook)
        self.current_log_dir: Optional[Path] = None
        self.events: List[GameEvent] = []
        self.log_manager: Optional[LogManager] = None
        
        self._setup_ui()

    def _setup_ui(self):
        """UI 구성"""
        # 상단: 디렉토리 선택 영역
        top_frame = ttk.Frame(self.frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="로그 디렉토리:").pack(side=tk.LEFT)
        
        self.path_var = tk.StringVar(value="선택된 디렉토리 없음")
        path_label = ttk.Label(top_frame, textvariable=self.path_var, relief=tk.SUNKEN)
        path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_select = ttk.Button(top_frame, text="디렉토리 선택", command=self._select_directory)
        btn_select.pack(side=tk.LEFT, padx=5)

        btn_refresh = ttk.Button(top_frame, text="새로고침", command=self._load_logs)
        btn_refresh.pack(side=tk.LEFT)

        # 필터 프레임
        filter_frame = ttk.LabelFrame(self.frame, text="필터", padding=10)
        filter_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        # Day 필터
        ttk.Label(filter_frame, text="Day:").pack(side=tk.LEFT, padx=(0, 5))
        self.day_var = tk.StringVar(value="전체")
        self.day_combo = ttk.Combobox(filter_frame, textvariable=self.day_var, width=10, state="readonly")
        self.day_combo["values"] = ["전체"]
        self.day_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.day_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

        # Phase 필터
        ttk.Label(filter_frame, text="Phase:").pack(side=tk.LEFT, padx=(0, 5))
        self.phase_var = tk.StringVar(value="전체")
        self.phase_combo = ttk.Combobox(filter_frame, textvariable=self.phase_var, width=15, state="readonly")
        self.phase_combo["values"] = ["전체", "낮 토론", "투표", "처형 여부 결정", "밤"]
        self.phase_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.phase_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

        # 이벤트 타입 필터
        ttk.Label(filter_frame, text="이벤트 타입:").pack(side=tk.LEFT, padx=(0, 5))
        self.event_type_var = tk.StringVar(value="전체")
        self.event_type_combo = ttk.Combobox(filter_frame, textvariable=self.event_type_var, width=15, state="readonly")
        self.event_type_combo["values"] = ["전체", "주장", "투표", "처형", "살해", "보호", "조사"]
        self.event_type_combo.pack(side=tk.LEFT)
        self.event_type_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

        # 중앙: 로그 표시 영역 (ScrolledText)
        log_frame = ttk.LabelFrame(self.frame, text="게임 이벤트 로그", padding=10)
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=("맑은 고딕", 10),
            bg="#f5f5f5",
            fg="#000000",
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 태그 설정 (색상 구분)
        self.log_text.tag_config("header", font=("맑은 고딕", 11, "bold"), foreground="#0066cc")
        self.log_text.tag_config("event", font=("맑은 고딕", 10), foreground="#333333")
        self.log_text.tag_config("vote", foreground="#ff6600")
        self.log_text.tag_config("execute", foreground="#cc0000", font=("맑은 고딕", 10, "bold"))
        self.log_text.tag_config("kill", foreground="#990000")
        self.log_text.tag_config("protect", foreground="#009900")
        self.log_text.tag_config("investigate", foreground="#0000cc")

        # 하단: 통계 정보
        stats_frame = ttk.LabelFrame(self.frame, text="통계", padding=10)
        stats_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))

        self.stats_var = tk.StringVar(value="이벤트 로드되지 않음")
        stats_label = ttk.Label(stats_frame, textvariable=self.stats_var)
        stats_label.pack()

    def _select_directory(self):
        """로그 디렉토리 선택"""
        from tkinter import filedialog
        
        directory = filedialog.askdirectory(
            title="로그 디렉토리 선택",
            initialdir="./logs"
        )
        
        if directory:
            self.current_log_dir = Path(directory)
            self.path_var.set(str(self.current_log_dir))
            self._load_logs()

    def _load_logs(self):
        """events.jsonl 파일에서 로그 로드"""
        if not self.current_log_dir:
            self._show_message("디렉토리를 먼저 선택해주세요.")
            return

        jsonl_path = self.current_log_dir / "events.jsonl"
        if not jsonl_path.exists():
            self._show_message(f"events.jsonl 파일을 찾을 수 없습니다: {jsonl_path}")
            return

        # LogManager 초기화 (interpret_event 사용을 위해)
        try:
            # 더미 LogManager 생성 (이미 로그가 있으므로 기록은 안 함)
            self.log_manager = LogManager(
                experiment_name="viewer",
                log_dir=str(self.current_log_dir.parent),
                use_tensorboard=False
            )
        except Exception as e:
            print(f"LogManager 초기화 실패 (interpret_event 없이 진행): {e}")
            self.log_manager = None

        # JSONL 파싱
        self.events = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        event = GameEvent(**data)
                        self.events.append(event)
        except Exception as e:
            self._show_message(f"로그 로드 실패: {e}")
            return

        # Day 필터 옵션 업데이트
        days = sorted(set(e.day for e in self.events))
        self.day_combo["values"] = ["전체"] + [f"Day {d}" for d in days]

        # 로그 표시
        self._apply_filter()

    def _apply_filter(self):
        """필터 적용하여 로그 표시"""
        if not self.events:
            return

        # 필터 값 가져오기
        day_filter = self.day_var.get()
        phase_filter = self.phase_var.get()
        event_type_filter = self.event_type_var.get()

        # 필터링
        filtered_events = []
        for event in self.events:
            # Day 필터
            if day_filter != "전체":
                day_num = int(day_filter.split()[1])
                if event.day != day_num:
                    continue

            # Phase 필터
            if phase_filter != "전체":
                phase_korean = self._phase_to_korean(event.phase)
                if phase_korean != phase_filter:
                    continue

            # 이벤트 타입 필터
            if event_type_filter != "전체":
                event_type_korean = self._event_type_to_korean(event.event_type)
                if event_type_korean != event_type_filter:
                    continue

            filtered_events.append(event)

        # 로그 표시
        self._display_logs(filtered_events)

        # 통계 업데이트
        self._update_stats(filtered_events)

    def _display_logs(self, events: List[GameEvent]):
        """필터링된 이벤트를 텍스트 위젯에 표시"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)

        if not events:
            self.log_text.insert(tk.END, "필터 조건에 맞는 이벤트가 없습니다.")
            self.log_text.config(state=tk.DISABLED)
            return

        # Day/Phase별로 그룹화
        grouped = defaultdict(list)
        for event in events:
            key = (event.day, event.phase)
            grouped[key].append(event)

        # 표시
        for (day, phase), group_events in sorted(grouped.items()):
            # 헤더
            header_text = f"\n═══ Day {day} - {self._phase_to_korean(phase)} ═══\n"
            self.log_text.insert(tk.END, header_text, "header")

            # 이벤트들
            for event in group_events:
                event_text = self._format_event(event)
                tag = self._get_event_tag(event.event_type)
                self.log_text.insert(tk.END, f"  • {event_text}\n", tag)

        self.log_text.config(state=tk.DISABLED)

    def _format_event(self, event: GameEvent) -> str:
        """GameEvent를 자연어 문장으로 변환 (LogManager 사용)"""
        # LogManager의 interpret_event 사용
        if self.log_manager:
            try:
                return self.log_manager.interpret_event(event)
            except Exception as e:
                print(f"interpret_event 실패, fallback 사용: {e}")
        
        # Fallback: 직접 변환
        event_type = event.event_type
        actor = f"Player {event.actor_id}" if event.actor_id != -1 else "시스템"
        target = f"Player {event.target_id}" if event.target_id != -1 else "없음"

        if event_type == EventType.CLAIM:
            role_name = Role(event.value).name if event.value is not None else "알 수 없음"
            if event.actor_id == event.target_id:
                return f"{actor}가 자신이 {role_name}이라고 주장했습니다."
            else:
                return f"{actor}가 {target}을(를) {role_name}(이)라고 주장했습니다."

        elif event_type == EventType.VOTE:
            return f"{actor}가 {target}에게 투표했습니다."

        elif event_type == EventType.EXECUTE:
            role_name = Role(event.value).name if event.value is not None else "알 수 없음"
            return f"{target}({role_name})이(가) 처형되었습니다."

        elif event_type == EventType.KILL:
            return f"{actor}가 {target}을(를) 살해 대상으로 지목했습니다."

        elif event_type == EventType.PROTECT:
            return f"{actor}가 {target}을(를) 보호했습니다."

        elif event_type == EventType.POLICE_RESULT:
            role_name = Role(event.value).name if event.value is not None else "알 수 없음"
            return f"{actor}가 {target}을(를) 조사한 결과 {role_name}입니다."

        else:
            return f"알 수 없는 이벤트: {event_type.name}"

    def _phase_to_korean(self, phase: Phase) -> str:
        """Phase enum을 한글로 변환"""
        phase_map = {
            Phase.DAY_DISCUSSION: "낮 토론",
            Phase.DAY_VOTE: "투표",
            Phase.DAY_EXECUTE: "처형 여부 결정",
            Phase.NIGHT: "밤",
        }
        return phase_map.get(phase, phase.name)

    def _event_type_to_korean(self, event_type: EventType) -> str:
        """EventType enum을 한글로 변환"""
        type_map = {
            EventType.CLAIM: "주장",
            EventType.VOTE: "투표",
            EventType.EXECUTE: "처형",
            EventType.KILL: "살해",
            EventType.PROTECT: "보호",
            EventType.POLICE_RESULT: "조사",
        }
        return type_map.get(event_type, event_type.name)

    def _get_event_tag(self, event_type: EventType) -> str:
        """이벤트 타입에 맞는 태그 반환"""
        tag_map = {
            EventType.VOTE: "vote",
            EventType.EXECUTE: "execute",
            EventType.KILL: "kill",
            EventType.PROTECT: "protect",
            EventType.POLICE_RESULT: "investigate",
        }
        return tag_map.get(event_type, "event")

    def _update_stats(self, events: List[GameEvent]):
        """통계 정보 업데이트"""
        total = len(events)
        if total == 0:
            self.stats_var.set("이벤트 없음")
            return

        # 이벤트 타입별 카운트
        type_counts = defaultdict(int)
        for event in events:
            type_counts[event.event_type] += 1

        stats_parts = [f"총 이벤트: {total}"]
        for event_type, count in type_counts.items():
            korean_name = self._event_type_to_korean(event_type)
            stats_parts.append(f"{korean_name}: {count}")

        self.stats_var.set(" | ".join(stats_parts))

    def _show_message(self, message: str):
        """메시지 표시"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, message)
        self.log_text.config(state=tk.DISABLED)

    def refresh(self):
        """탭 새로고침 (외부 호출용)"""
        if self.current_log_dir:
            self._load_logs()
