import json
import os
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path

from config import Role

class ExpertDataManager:
    def __init__(self, save_dir: Path):
        """
        save_dir: LogManager가 생성한 session_dir (Path 객체)
        """
        self.save_dir = save_dir
        self.save_path = self.save_dir / "train_set.jsonl"
        
        # 단일 환경이므로 복잡한 버퍼링 없이 바로 리스트에 담았다가 flush해도 됨
        # 하지만 확장성을 위해 episode_id 구조는 유지
        self.episode_buffers: Dict[int, Dict[int, Any]] = {}
        
        # 파일 초기화 (없으면 생성)
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def record_turn(self, episode_id: int, player_id: int, observation: Any, action: Any, action_mask: Optional[List[int]] = None):
        """한 턴의 (Observation, Action) 쌍을 기록"""
        
        # 1. 버퍼 초기화
        if episode_id not in self.episode_buffers:
            # [수정] masks 필드 추가
            self.episode_buffers[episode_id] = {i: {'obs': [], 'acts': [], 'masks': []} for i in range(8)}

        # 2. Observation 처리 강화
        # runner에서 np.array를 넘겨주겠지만, 혹시 모를 안전장치
        if isinstance(observation, np.ndarray):
            obs_list = observation.tolist()
        elif isinstance(observation, list):
            obs_list = observation
        else:
            # 딕셔너리가 넘어왔을 경우 처리 (fallback)
            if isinstance(observation, dict) and 'observation' in observation:
                obs = observation['observation']
                obs_list = obs.tolist() if isinstance(obs, np.ndarray) else list(obs)
            else:
                # print(f"[Warning] Unknown observation type: {type(observation)}")
                obs_list = []

        # 3. Action 처리 (GameAction 클래스와 일관성 유지)
        # state.py: Target -1 -> 0, 0~7 -> 1~8
        target_idx = 0 
        role_idx = 0
        
        # 객체(GameAction)인 경우
        if hasattr(action, 'to_multi_discrete'):
             vec = action.to_multi_discrete()
             target_idx = vec[0]
             role_idx = vec[1]
             
        # 벡터([target, role])인 경우
        elif isinstance(action, (list, np.ndarray, tuple)):
            target_idx = int(action[0])
            role_idx = int(action[1])

        # 4. 버퍼에 추가
        self.episode_buffers[episode_id][player_id]['obs'].append(obs_list)
        self.episode_buffers[episode_id][player_id]['acts'].append([target_idx, role_idx])
        
        if action_mask is not None:
             # Mask 처리 (Numpy -> List)
             mask_list = action_mask.tolist() if isinstance(action_mask, np.ndarray) else list(action_mask)
             self.episode_buffers[episode_id][player_id]['masks'].append(mask_list)

    def flush_episode(self, episode_id: int, winner_role: Optional['Role'] = None, players: List[Any] = None):
        """
        에피소드 종료 시 파일에 저장 (Append)
        Args:
            episode_id: 에피소드 ID
            winner_role: 승리한 팀 (Role.MAFIA 또는 Role.CITIZEN)
            players: 플레이어 객체 리스트 (역할 확인용)
        """
        if episode_id not in self.episode_buffers:
            return

        buffer = self.episode_buffers[episode_id]
        
        try:
            with open(self.save_path, 'a', encoding='utf-8') as f:
                for p_id in range(8):
                    # 데이터가 없으면 스킵
                    if len(buffer[p_id]['obs']) == 0:
                        continue
                        
                    # [필터링 로직 추가] 
                    # players 정보가 넘어왔다면, 승리한 팀의 데이터인지 확인
                    is_winner = False
                    if players and winner_role is not None:
                        player_role = players[p_id].role
                        # 마피아 승리 시: 마피아 팀만 저장
                        if winner_role == 3: # Role.MAFIA
                            if player_role == 3: is_winner = True
                        # 시민 승리 시: 시민 팀(시민, 경찰, 의사)만 저장
                        else: 
                            if player_role != 3: is_winner = True
                    
                    # 승리한 데이터만 저장하고 싶다면 아래 주석 해제
                    # if not is_winner: continue

                    entry = {
                        "episode_id": episode_id,
                        "player_id": p_id,
                        "role": int(players[p_id].role) if players else -1, # 역할 정보 추가
                        "is_winner": is_winner, # 나중에 데이터 로더에서 필터링 가능하도록 태깅
                        "obs": buffer[p_id]['obs'],
                        "acts": buffer[p_id]['acts']
                    }
                    # mask가 있으면 함께 저장
                    if len(buffer[p_id]['masks']) > 0:
                            entry["masks"] = buffer[p_id]['masks']
                            
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[DataManager Error] Flush failed: {e}")
        
        # 메모리 해제
        del self.episode_buffers[episode_id]