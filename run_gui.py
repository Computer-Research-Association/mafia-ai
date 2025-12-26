import tkinter as tk
from gui.gui_viewer import MafiaLogViewerApp  # gui 폴더 안의 app.py에서 클래스 가져오기

if __name__ == "__main__":
    root = tk.Tk()
    app = MafiaLogViewerApp(root)
    root.mainloop()
