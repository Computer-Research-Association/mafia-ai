import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.log_parser import parse_game_logs


class AIStatsTab:
    def __init__(self, parent, log_path):
        self.frame = ttk.Frame(parent)
        self.log_path = log_path
        self._init_ui()

    def _init_ui(self):
        btn_refresh = ttk.Button(self.frame, text="새로고침", command=self.refresh)
        btn_refresh.pack(side=tk.TOP, anchor=tk.E, pady=(0, 10))
        self.chart_frame = ttk.Frame(self.frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

    def refresh(self):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        games = parse_game_logs(self.log_path)

        if not games:
            ttk.Label(self.chart_frame, text="데이터가 없습니다.").pack()
            return

        CHUNK_SIZE = 100
        x_labels, y_values = [], []
        total_games = len(games)

        for i in range(0, total_games, CHUNK_SIZE):
            chunk = games[i : i + CHUNK_SIZE]
            if not chunk:
                continue
            wins, valid = 0, 0
            for g in chunk:
                ai_won = g.get("ai_won")
                if ai_won is None:
                    continue
                valid += 1
                if ai_won:
                    wins += 1

            rate = (wins / valid * 100) if valid > 0 else 0
            x_labels.append(f"{i+1}\n~\n{min(i+CHUNK_SIZE, total_games)}")
            y_values.append(rate)

        self._draw_chart(x_labels, y_values)

    def _draw_chart(self, x, y):
        if not x:
            return
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        bars = ax.bar(x, y, color="#66b3ff", edgecolor="white")
        ax.set_ylim(0, 100)
        ax.set_title("구간별 AI 승률 변화")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
