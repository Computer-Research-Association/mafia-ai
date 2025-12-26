import os
import re


def parse_game_logs(file_path):
    """로그 파일을 읽어 게임 데이터를 리스트로 반환"""
    games = []
    if not os.path.exists(file_path):
        return games

    current_game = {"winner": None, "roles": {}, "Day": 0, "ai_won": None}
    in_episode = False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Episode" in line and "Start" in line:
                    in_episode = True
                    current_game = {
                        "winner": None,
                        "roles": {},
                        "Day": 0,
                        "ai_won": None,
                    }
                    continue

                if not in_episode:
                    continue

                if "Winner" in line:
                    if "Mafia" in line:
                        current_game["winner"] = "Mafia"
                    elif "Citizen" in line:
                        current_game["winner"] = "Citizen"

                if "Player 0" in line or "Player0" in line:
                    upper_line = line.upper()
                    if "MAFIA" in upper_line:
                        current_game["roles"][0] = "MAFIA"
                    elif "POLICE" in upper_line:
                        current_game["roles"][0] = "POLICE"
                    elif "DOCTOR" in upper_line:
                        current_game["roles"][0] = "DOCTOR"
                    elif "CITIZEN" in upper_line:
                        current_game["roles"][0] = "CITIZEN"

                day_match = re.search(r"Day\s*[:]?\s*(\d+)", line, re.IGNORECASE)
                if day_match:
                    d = int(day_match.group(1))
                    if d > current_game["Day"]:
                        current_game["Day"] = d

                if "Episode" in line and "End" in line:
                    win_match = re.search(r"Win:\s*(True|False)", line, re.IGNORECASE)
                    if win_match:
                        current_game["ai_won"] = win_match.group(1).lower() == "true"

                    if current_game["Day"] == 0:
                        current_game["Day"] = 1

                    games.append(current_game)
                    in_episode = False
    except Exception as e:
        print(f"[Error] 파싱 중 오류 발생: {e}")

    return games


def calculate_avg_days(file_path):
    """평균 진행 일수 계산"""
    if not os.path.exists(file_path):
        return 0.0

    total_days_sum = 0
    episode_count = 0
    current_max_day = 0
    in_episode = False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Episode" in line and "Start" in line:
                    in_episode = True
                    current_max_day = 0

                if in_episode:
                    day_match = re.search(r"Day\s*(\d+)", line, re.IGNORECASE)
                    if day_match:
                        day_num = int(day_match.group(1))
                        if day_num > current_max_day:
                            current_max_day = day_num

                if "Episode" in line and "End" in line:
                    if in_episode:
                        total_days_sum += max(1, current_max_day)
                        episode_count += 1
                    in_episode = False

        if episode_count == 0:
            return 0.0
        return total_days_sum / episode_count
    except:
        return 0.0
