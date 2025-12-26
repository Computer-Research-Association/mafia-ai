import os
import platform
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt


# 1. 폰트 설정
def configure_fonts():
    if platform.system() == "Windows":
        try:
            font_name = font_manager.FontProperties(
                fname="c:/Windows/Fonts/malgun.ttf"
            ).get_name()
            rc("font", family=font_name)
        except:
            pass
    elif platform.system() == "Darwin":
        rc("font", family="AppleGothic")
    else:
        rc("font", family="NanumGothic")

    plt.rcParams["axes.unicode_minus"] = False


# 2. 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # mafia-ai/gui
BASE_DIR = os.path.dirname(CURRENT_DIR)  # mafia-ai/
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE_NAME = "mafia_game_log.txt"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)
