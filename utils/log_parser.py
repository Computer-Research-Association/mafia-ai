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
                # 1. 에피소드 시작
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

                # 2. 승자 판별
                if "Winner" in line:
                    if "Mafia" in line:
                        current_game["winner"] = "Mafia"
                    elif "Citizen" in line:
                        current_game["winner"] = "Citizen"

                # 3. AI 역할 판별 (한글/영어 지원)
                if any(k in line for k in ["Player 0", "Player0", "플레이어 0"]):
                    upper = line.upper()
                    if "MAFIA" in upper:
                        current_game["roles"][0] = "MAFIA"
                    elif "POLICE" in upper:
                        current_game["roles"][0] = "POLICE"
                    elif "DOCTOR" in upper:
                        current_game["roles"][0] = "DOCTOR"
                    elif "CITIZEN" in upper:
                        current_game["roles"][0] = "CITIZEN"

                # 4. Day 추출
                day_match = re.search(r"Day\s*[:]?\s*(\d+)", line, re.IGNORECASE)
                if day_match:
                    d = int(day_match.group(1))
                    if d > current_game["Day"]:
                        current_game["Day"] = d

                # 5. 종료 및 승패 파싱
                if "Episode" in line and "End" in line:
                    win_match = re.search(r"Win:\s*(True|False)", line, re.IGNORECASE)
                    if win_match:
                        current_game["ai_won"] = win_match.group(1).lower() == "true"

                    if current_game["Day"] == 0:
                        current_game["Day"] = 1
                    games.append(current_game)
                    in_episode = False
    except Exception as e:
        print(f"[LogParser Error] 파싱 중 오류: {e}")

    return games


# [신규 추가] 승률 계산 로직을 여기로 이동!
def calculate_win_stats(games):
    """게임 리스트를 받아 팀별 승수와 AI 직업별 전적을 계산하여 반환"""
    team_wins = {"MAFIA": 0, "CITIZEN": 0}
    ai_stats = {"m_play": 0, "m_win": 0, "c_play": 0, "c_win": 0}

    for game in games:
        winner = game.get("winner")
        ai_role = game.get("roles", {}).get(0)
        ai_won = game.get("ai_won")

        # 승자 정보가 없으면 AI 승패로 추론 (보정 로직)
        if winner is None and ai_role and (ai_won is not None):
            if ai_role == "MAFIA":
                winner = "Mafia" if ai_won else "Citizen"
            else:
                winner = "Citizen" if ai_won else "Mafia"

        # 1. 팀 승리 집계
        if winner == "Mafia":
            team_wins["MAFIA"] += 1
        elif winner == "Citizen":
            team_wins["CITIZEN"] += 1

        # 2. AI 상세 전적 집계
        if ai_role and (ai_won is not None):
            if ai_role == "MAFIA":
                ai_stats["m_play"] += 1
                if ai_won:
                    ai_stats["m_win"] += 1
            else:
                ai_stats["c_play"] += 1
                if ai_won:
                    ai_stats["c_win"] += 1

    return team_wins, ai_stats


def calculate_avg_days(file_path):
    """평균 진행 일수 계산"""
    # ... (기존과 동일하므로 생략하지 않고 안전하게 포함) ...
    if not os.path.exists(file_path):
        return 0.0
    total = 0
    count = 0
    curr_max = 0
    in_ep = False
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Episode" in line and "Start" in line:
                    in_ep = True
                    curr_max = 0
                if in_ep:
                    dm = re.search(r"Day\s*(\d+)", line, re.IGNORECASE)
                    if dm:
                        d = int(dm.group(1))
                        if d > curr_max:
                            curr_max = d
                if "Episode" in line and "End" in line:
                    if in_ep:
                        total += max(1, curr_max)
                        count += 1
                    in_ep = False
        return total / count if count > 0 else 0.0
    except:
        return 0.0
