import sys
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# 모듈 임포트
from .gui_config import configure_fonts, LOG_FILE_PATH
from .tabs.team_stats import TeamStatsTab
from .tabs.ai_stats import AIStatsTab
from .tabs.replay import ReplayTab


class MafiaLogViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mafia AI 통합 뷰어")
        self.root.geometry("1100x750")

        # 폰트 설정 적용
        configure_fonts()

        # 탭 컨트롤 생성
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=1, fill="both")

        # 각 탭 초기화 (탭 클래스에 탭이 그려질 Frame을 넘겨주지 않고,
        # 탭 클래스 내부에서 Frame을 생성하여 Notebook에 add 하는 방식)

        # 1. 팀별 승률 탭
        self.tab1 = TeamStatsTab(self.notebook, LOG_FILE_PATH)
        self.notebook.add(self.tab1.frame, text="팀별 승률")
        self.tab1.refresh()  # 초기 로드

        # 2. AI 성장 그래프 탭
        self.tab2 = AIStatsTab(self.notebook, LOG_FILE_PATH)
        self.notebook.add(self.tab2.frame, text="AI 성장 그래프")
        self.tab2.refresh()  # 초기 로드

        # 3. 리플레이 탭
        self.tab3 = ReplayTab(self.notebook, LOG_FILE_PATH)
        self.notebook.add(self.tab3.frame, text="에피소드 리플레이")

        # 종료 이벤트 연결
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        try:
            plt.close("all")
            self.root.quit()
            self.root.destroy()
        except:
            pass
        finally:
            sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = MafiaLogViewerApp(root)
    root.mainloop()
