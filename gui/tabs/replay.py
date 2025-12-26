import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import re


class ReplayTab:
    def __init__(self, parent, log_path):
        self.frame = ttk.Frame(parent)
        self.log_path = log_path
        self.episode_start_pos = {}
        self._init_ui()
        self.refresh_list()

    def _init_ui(self):
        paned = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left = ttk.Frame(paned, width=250)
        paned.add(left, weight=1)

        header = ttk.Frame(left)
        header.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Label(header, text="에피소드 목록").pack(side=tk.LEFT)
        ttk.Button(header, text="새로고침", width=8, command=self.refresh_list).pack(
            side=tk.RIGHT
        )

        scroll = ttk.Scrollbar(left)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(
            left, yscrollcommand=scroll.set, font=("Consolas", 11)
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        right = ttk.Frame(paned)
        paned.add(right, weight=4)
        ttk.Label(right, text="게임 로그").pack(pady=5)
        self.text_area = scrolledtext.ScrolledText(
            right, font=("맑은 고딕", 10), state="disabled"
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)

    def refresh_list(self):
        self.episode_start_pos = {}
        self.listbox.delete(0, tk.END)
        if not os.path.exists(self.log_path):
            return

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    match = re.search(r"Episode (\d+) Start", line)
                    if match:
                        ep = int(match.group(1))
                        self.episode_start_pos[ep] = pos
                        self.listbox.insert(tk.END, f"Episode {ep}")
        except Exception as e:
            messagebox.showerror("오류", f"인덱싱 오류: {e}")

    def on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        ep = int(self.listbox.get(sel[0]).split()[-1])
        start = self.episode_start_pos.get(ep)

        lines = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                f.seek(start)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
                    if f"Episode {ep} End" in line:
                        break

            self.text_area.config(state="normal")
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, "".join(lines))
            self.text_area.see(1.0)
            self.text_area.config(state="disabled")
        except:
            pass
