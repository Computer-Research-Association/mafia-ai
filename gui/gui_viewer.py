import sys
import tkinter as tk
from tkinter import ttk

# 모듈 임포트
from .gui_config import configure_fonts, LOG_FILE_PATH
from .tabs.log_viewer import LogViewerTab


class MafiaLogViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mafia AI 게임 로그 뷰어")
        self.root.geometry("1100x750")

        # 폰트 설정 적용
        configure_fonts()

        # 게임 로그 탭만 사용
        self.log_viewer = LogViewerTab(root, LOG_FILE_PATH)
        self.log_viewer.frame.pack(fill=tk.BOTH, expand=True)

        # 종료 이벤트 연결
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        try:
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
