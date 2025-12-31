import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import os
import json
from pathlib import Path
from typing import List, Optional

from state import GameEvent
from core.logger import LogManager


class ReplayTab:
    """
    게임 리플레이 탭 (JSONL 기반)
    
    LogManager의 interpret_event()를 사용하여 일관된 내러티브를 표시합니다.
    """
    
    def __init__(self, parent, log_dir: str = "./logs"):
        self.frame = ttk.Frame(parent)
        self.log_dir = Path(log_dir)
        self.current_session_dir: Optional[Path] = None
        self.events: List[GameEvent] = []
        
        # 임시 LogManager (해석 전용)
        self.temp_logger: Optional[LogManager] = None

        self._init_ui()
        self.refresh_sessions()

    def _init_ui(self):
        """UI 초기화"""
        paned = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 왼쪽: 세션 목록
        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=1)

        header = ttk.Frame(left)
        header.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Label(header, text="실험 세션").pack(side=tk.LEFT)
        ttk.Button(header, text="새로고침", width=8, command=self.refresh_sessions).pack(
            side=tk.RIGHT
        )
        ttk.Button(header, text="열기", width=8, command=self.open_custom_log).pack(
            side=tk.RIGHT, padx=5
        )

        scroll = ttk.Scrollbar(left)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(
            left, yscrollcommand=scroll.set, font=("Consolas", 10)
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_session)

        # 오른쪽: 이벤트 로그
        right = ttk.Frame(paned)
        paned.add(right, weight=4)

        ttk.Label(right, text="게임 이벤트 리플레이").pack(pady=5)

        self.text_area = scrolledtext.ScrolledText(
            right, font=("맑은 고딕", 10), state="disabled"
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)

    def refresh_sessions(self):
        """로그 디렉토리에서 실험 세션 목록 갱신"""
        self.listbox.delete(0, tk.END)
        
        if not self.log_dir.exists():
            return

        # 디렉토리 목록 (타임스탬프 역순)
        sessions = sorted(
            [d for d in self.log_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True
        )

        for session_dir in sessions:
            jsonl_path = session_dir / "events.jsonl"
            if jsonl_path.exists():
                self.listbox.insert(tk.END, session_dir.name)

    def open_custom_log(self):
        """사용자가 직접 JSONL 파일 선택"""
        file_path = filedialog.askopenfilename(
            title="JSONL 로그 파일 선택",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_session_dir = Path(file_path).parent
            self.load_and_display_events(Path(file_path))

    def on_select_session(self, event):
        """세션 선택 시 이벤트 로드"""
        sel = self.listbox.curselection()
        if not sel:
            return
        
        session_name = self.listbox.get(sel[0])
        session_dir = self.log_dir / session_name
        jsonl_path = session_dir / "events.jsonl"
        
        self.current_session_dir = session_dir
        self.load_and_display_events(jsonl_path)

    def load_and_display_events(self, jsonl_path: Path):
        """JSONL 파일에서 이벤트를 로드하고 내러티브로 변환하여 표시"""
        try:
            # 이벤트 로드
            self.events = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        event_dict = json.loads(line)
                        self.events.append(GameEvent(**event_dict))

            # 임시 LogManager 생성 (해석용)
            if self.temp_logger:
                self.temp_logger.close()
            self.temp_logger = LogManager(
                experiment_name="temp_replay",
                log_dir=str(self.current_session_dir)
            )

            # 내러티브 변환
            narratives = []
            narratives.append(f"=== 리플레이: {jsonl_path.name} ===\n")
            narratives.append(f"총 {len(self.events)}개의 이벤트\n")
            narratives.append("=" * 50 + "\n\n")

            for event in self.events:
                narrative = self.temp_logger.interpret_event(event)
                narratives.append(f"{narrative}\n")

            # 텍스트 영역에 표시
            self.text_area.config(state="normal")
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, "".join(narratives))
            self.text_area.see(1.0)
            self.text_area.config(state="disabled")

        except Exception as e:
            messagebox.showerror("오류", f"로그 로드 실패: {e}")
            print(f"Error loading JSONL: {e}")
    
    def __del__(self):
        """리소스 정리"""
        if self.temp_logger:
            try:
                self.temp_logger.close()
            except:
                pass

