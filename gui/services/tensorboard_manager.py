import socket
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Optional


class TensorBoardManager:
    def __init__(self, port: int = 6006):
        self.process: Optional[subprocess.Popen] = None
        self.port = port

    def is_port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", self.port)) == 0

    def shutdown(self):
        """실행 중인 텐서보드 프로세스 종료"""
        # 1. 파이썬 객체로 관리 중인 프로세스 종료
        if self.process:
            self.process.terminate()
            self.process = None

        # 2. (Windows) 포트 점유 중인 tensorboard.exe 강제 종료
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "tensorboard.exe", "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True,
            )
        except Exception as e:
            print(f"Error killing tensorboard: {e}")

    def launch(self, log_dir: Path) -> bool:
        """텐서보드 실행 및 브라우저 오픈"""
        self.shutdown()  # 기존 프로세스 정리

        cmd = [
            "tensorboard",
            "--logdir",
            str(log_dir),
            "--port",
            str(self.port),
            "--reload_interval",
            "5",
        ]

        try:
            creationflags = 0
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                creationflags = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                cmd,
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )

            # 서비스 시작 대기 (최대 15초)
            if self._wait_for_service():
                webbrowser.open(f"http://localhost:{self.port}")
                return True
            else:
                print("TensorBoard failed to start within timeout.")
                return False

        except Exception as e:
            print(f"TensorBoard Launch Error: {e}")
            return False

    def _wait_for_service(self, timeout=15) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_port_in_use():
                return True
            time.sleep(0.5)
        return False
