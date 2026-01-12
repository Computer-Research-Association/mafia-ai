import os
import json
import yaml
import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING, Any
from openai import OpenAI
from dotenv import load_dotenv
import random
from core.agents.base_agent import BaseAgent
from config import config, Role, Phase, EventType, ActionType
from core.engine.state import GameStatus, GameEvent, GameAction
from collections import defaultdict

if TYPE_CHECKING:
    from core.managers.logger import LogManager

load_dotenv()


class LLMAgent(BaseAgent):
    def __init__(
        self,
        player_id: int,
        role: Role = Role.CITIZEN,
        logger: Optional["LogManager"] = None,
    ):
        super().__init__(player_id, role)

        # LogManager 인스턴스 (내러티브 해석용)
        self.logger = logger

        # API 설정
        self.client = OpenAI(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar",
        )
        self.model = "solar-pro"

        # Load prompts
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.yaml_data = yaml.safe_load(f)

        self.phase_schemas = self.yaml_data.get("PHASE_SCHEMAS", {})
        self.json_format_block = self.yaml_data.get("JSON_FORMAT_BLOCK", "")
        self.game_data_block = self.yaml_data.get("GAME_DATA_BLOCK", "")

    def _phase_to_korean(self, phase: Phase) -> str:
        """Phase enum을 한글로 변환"""
        phase_map = {
            Phase.GAME_START: "게임 시작",
            Phase.DAY_DISCUSSION: "낮 토론",
            Phase.DAY_VOTE: "투표",
            Phase.DAY_EXECUTE: "처형 여부 결정",
            Phase.NIGHT: "밤",
            Phase.GAME_END: "게임 종료",
        }
        return phase_map.get(phase, phase.name)

    def _group_events_by_day_phase(self, events: List[GameEvent]) -> dict:
        """이벤트를 day와 phase로 그룹화"""
        from collections import defaultdict

        grouped = defaultdict(list)
        for event in events:
            key = (event.day, event.phase)
            grouped[key].append(event)
        return dict(sorted(grouped.items()))

    def _format_event(self, event: GameEvent) -> str:
        """GameEvent를 logger.py의 interpret_event를 사용하여 자연어 문장으로 변환"""
        if self.logger:
            return self.logger.interpret_event(event)
        return f"LogManager not available to interpret event: {event}"

    def _format_game_status(self, status: GameStatus) -> str:
        """
        GameStatus를 LLM이 이해하기 쉬운 형태로 변환합니다.

        Returns:
            자연어 형식의 게임 상태 설명
        """
        lines = []

        # 기본 정보
        lines.append(f"### 현재 게임 상태")
        lines.append(f"- Day {status.day}, {self._phase_to_korean(status.phase)}")
        lines.append("")

        # 생존자 목록
        alive_players = [p.id for p in status.players if p.alive]
        dead_players = [p.id for p in status.players if not p.alive]
        lines.append(
            f"### 생존자: {', '.join(f'Player {pid}' for pid in alive_players)}"
        )
        if dead_players:
            lines.append(
                f"### 사망자: {', '.join(f'Player {pid}' for pid in dead_players)}"
            )
        lines.append("")

        # 이벤트 히스토리 (자연어 변환)
        lines.append("### 게임 이벤트 기록")
        if not status.action_history:
            lines.append("아직 이벤트가 없습니다.")
        else:
            grouped_events = self._group_events_by_day_phase(status.action_history)
            for (day, phase), events in grouped_events.items():
                lines.append(f"\n**Day {day}, {self._phase_to_korean(phase)}:**")
                for event in events:
                    lines.append(f"  - {self._format_event(event)}")

        return "\n".join(lines)

    def translate_to_engine(self, action_dict: Dict[str, Any]) -> GameAction:
        """
        LLM JSON 응답을 MafiaAction으로 변환하는 순수 번역기

        딕셔너리 구조:
        - target_id: int (지목 대상, -1은 없음)
        - role: Optional[Role] (주장하는 역할)
        - discussion_status: str (토론 종료 여부, "End"면 PASS)

        Args:
            action_dict: LLM이 반환한 JSON 딕셔너리

        Returns:
            MafiaAction 객체
        """
        target_id = action_dict.get("target_id", -1)
        # target_id가 None인 경우 -1로 처리
        if target_id is None:
            target_id = -1
        role_str = action_dict.get("role")
        discussion_status = action_dict.get("discussion_status", "Continue")

        # 역할을 Role enum으로 변환 (정수 또는 문자열 처리)
        claim_role = None
        if role_str is not None:
            if isinstance(role_str, int):
                # 정수인 경우 (0:CITIZEN, 1:POLICE, 2:DOCTOR, 3:MAFIA)
                try:
                    claim_role = Role(role_str)
                except ValueError:
                    print(f"[Player {self.id}] Invalid role integer: {role_str}")
            elif isinstance(role_str, str):
                # 문자열인 경우
                role_upper = role_str.upper()
                if hasattr(Role, role_upper):
                    claim_role = Role[role_upper]
                else:
                    print(f"[Player {self.id}] Invalid role string: {role_str}")

        # 토론 종료 → PASS
        if discussion_status == "End":
            return GameAction(
                action_type=ActionType.PASS, target_id=-1, claim_role=None
            )

        # 역할 주장 (타겟 포함 여부 무관하게 CLAIM으로 통합)
        if claim_role is not None:
            return GameAction(
                action_type=ActionType.CLAIM, target_id=target_id, claim_role=claim_role
            )

        # 단순 지목
        if target_id != -1:
            return GameAction(
                action_type=ActionType.TARGET_ACTION,
                target_id=target_id,
                claim_role=None,
            )

        # 기권
        return GameAction(action_type=ActionType.PASS, target_id=-1, claim_role=None)

    def get_action(self, status: GameStatus) -> GameAction:
        """
        LLM의 JSON 응답을 GameAction으로 변환하여 반환
        """
        alive_status = next((p for p in status.players if p.id == self.id), None)
        if alive_status and not alive_status.alive:
            return GameAction(
                action_type=ActionType.PASS, target_id=-1, claim_role=None
            )
        self.role = status.my_role

        phase_name = status.phase.name
        role_name = self.role.name

        role_specific_key = f"{role_name}_{phase_name}"
        if role_specific_key in self.yaml_data:
            prompt_key = role_specific_key
        else:
            prompt_key = phase_name

        # LLM 실행 및 JSON 응답 파싱
        action_dict = self._execute_ai_logic(prompt_key, status)

        if status.phase == Phase.DAY_EXECUTE:
            # 1. 현재 처형 후보 찾기 (투표 기록 분석)
            vote_counts = defaultdict(int)
            for event in status.action_history:
                if (
                    event.day == status.day
                    and event.event_type == EventType.VOTE
                    and event.target_id != -1
                ):
                    vote_counts[event.target_id] += 1

            candidate_id = -1
            if vote_counts:
                candidate_id = max(vote_counts, key=vote_counts.get)

            # 2. LLM의 의도(agree_execution)를 target_id로 변환
            agree = int(action_dict.get("agree_execution", -1))

            if agree == 1 and candidate_id != -1:
                action_dict["target_id"] = candidate_id
            else:
                action_dict["target_id"] = -1

        # 일반 타겟 파싱
        raw_target = action_dict.get("target_id")
        if isinstance(raw_target, str) and raw_target.isdigit():
            action_dict["target_id"] = int(raw_target)

        # 액션 유효성 검증 및 보정
        if status.phase in [Phase.NIGHT, Phase.DAY_VOTE]:
            target_id = action_dict.get("target_id")
            alive_players = [p.id for p in status.players if p.alive]

            # 1. 경찰일 경우 이미 조사한 대상 목록 만들기
            investigated_players = set()
            if self.role == Role.POLICE:
                for event in status.action_history:
                    if (
                        event.event_type == EventType.POLICE_RESULT
                        and event.actor_id == self.id
                    ):
                        investigated_players.add(event.target_id)

            is_valid = True

            # 0. None 처리
            if target_id is None:
                target_id = -1

            # 1. 살아있는 플레이어를 타겟했는가?
            if target_id != -1 and target_id not in alive_players:
                is_valid = False

            # 2. (투표 시) 자기 자신에게 투표했는가?
            if status.phase == Phase.DAY_VOTE and target_id == self.id:
                is_valid = False

            # 3. (경찰) 자기 자신을 조사했거나, 이미 조사한 대상을 또 찍었는가?
            if self.role == Role.POLICE and status.phase == Phase.NIGHT:
                if target_id == self.id:
                    is_valid = False
                elif target_id in investigated_players:
                    is_valid = False

            # 4. (마피아) 동료 또는 자신을 공격했는가?
            if self.role == Role.MAFIA:
                mafia_team = {self.id}
                for event in status.action_history:
                    if (
                        event.event_type == EventType.POLICE_RESULT
                        and event.value == Role.MAFIA
                    ):
                        mafia_team.add(event.target_id)

                if target_id in mafia_team:
                    is_valid = False

            # 유효하지 않은 액션일 경우, 안전한 타겟으로 강제 변경
            if not is_valid:
                possible_targets = list(set(alive_players) - {self.id})

                # 마피아는 팀킬 방지
                if self.role == Role.MAFIA:
                    possible_targets = list(set(possible_targets) - mafia_team)

                # 경찰은 이미 조사한 사람 제외
                if self.role == Role.POLICE:
                    possible_targets = list(
                        set(possible_targets) - investigated_players
                    )

                if possible_targets:
                    action_dict["target_id"] = random.choice(possible_targets)
                else:
                    action_dict["target_id"] = -1

        return self.translate_to_engine(action_dict)

    def _execute_ai_logic(self, prompt_key: str, status: GameStatus) -> Dict[str, Any]:
        """LLM 실행 및 응답을 딕셔너리로 파싱하여 반환"""
        status_json = status.model_dump_json(exclude_none=True)
        phase_name = status.phase.name

        prompt_data = self.yaml_data.get(prompt_key)
        if not prompt_data:
            print(
                f"[Player {self.id}] Warning: Prompt key '{prompt_key}' not found, returning empty action"
            )
            # 프롬프트가 없으면 빈 target_id 반환 (PASS로 처리됨)
            return {"target_id": -1}

        role_specific_system_msg = prompt_data.get("system", "")
        json_schema = self.phase_schemas.get(
            prompt_key, self.phase_schemas.get(phase_name, {})
        )
        json_instruction = self.json_format_block.format(json_schema=json_schema)
        final_system_msg = f"{role_specific_system_msg}\n{json_instruction}"

        user_template = prompt_data.get("user", "")
        conversation_log_for_llm = self._create_conversation_log(status)
        structured_summary_for_llm = self._create_structured_summary(status)

        llm_status_lines = []
        llm_status_lines.append(
            f"- Day {status.day}, {self._phase_to_korean(status.phase)}"
        )
        alive_players = [p.id for p in status.players if p.alive]
        dead_players = [p.id for p in status.players if not p.alive]
        llm_status_lines.append(
            f"- 생존자: {', '.join(f'Player {pid}' for pid in alive_players)}"
        )
        if dead_players:
            llm_status_lines.append(
                f"- 사망자: {', '.join(f'Player {pid}' for pid in dead_players)}"
            )

        game_data_for_llm = self.game_data_block.format(
            role_name=self.role.name,
            id=self.id,
            game_status="\n".join(llm_status_lines),
            conversation_log=conversation_log_for_llm,
            structured_summary=structured_summary_for_llm,
        )

        final_user_msg = user_template.format(game_data=game_data_for_llm)

        # LLM 호출 및 JSON 파싱
        response_str = self._call_llm(final_system_msg, final_user_msg)
        try:
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            print(f"[Player {self.id}] Failed to parse LLM response: {e}")
            return {
                "error": "Invalid JSON response from LLM",
                "raw_response": response_str,
            }

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """LLM API 호출 - JSON 문자열 반환"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _create_conversation_log(self, status: GameStatus) -> str:
        """
        게임의 모든 공개 이벤트를 요약하여 대화 및 행동 기록을 생성합니다.
        LogManager의 interpret_event()를 사용하여 일관된 내러티브를 생성합니다.
        """
        if not self.logger:
            return "LogManager가 설정되지 않아 기록을 표시할 수 없습니다."

        # LogManager를 통한 통합 해석 (중복 제거)
        log_lines = [self.logger.interpret_event(e) for e in status.action_history]

        return "\n".join(log_lines) if log_lines else "아직 기록된 행동이 없습니다."

    def _create_structured_summary(self, status: GameStatus) -> str:
        """
        게임 기록을 분석하여 투표, 주장, 사망 등 핵심 정보를 요약한 텍스트를 생성합니다.
        """

        summary_lines = []
        vote_details = defaultdict(lambda: defaultdict(list))
        for event in status.action_history:
            if event.event_type == EventType.VOTE and event.target_id != -1:
                vote_details[event.day][event.target_id].append(event.actor_id)

        if vote_details:
            summary_lines.append("* 투표 요약:")
            for day, votes in sorted(vote_details.items()):
                day_summary = []
                sorted_targets = sorted(
                    votes.items(), key=lambda item: len(item[1]), reverse=True
                )
                for target_id, voters in sorted_targets:
                    voter_str = ", ".join(map(str, sorted(voters)))
                    day_summary.append(
                        f"Player {target_id} ({len(voters)}표) M-- [{voter_str}]"
                    )
                summary_lines.append(f"  - {day}일차: " + " | ".join(day_summary))

        if status.phase == Phase.DAY_EXECUTE:
            current_votes = vote_details.get(status.day, {})
            if current_votes:
                # 득표수 계산
                counts = {pid: len(voters) for pid, voters in current_votes.items()}
                if counts:
                    max_vote = max(counts.values())
                    candidates = [pid for pid, c in counts.items() if c == max_vote]

                    summary_lines.append("\n[현재 상황 알림]")
                    if len(candidates) == 1:
                        target = candidates[0]
                        summary_lines.append(
                            f"최다 득표자(처형 후보): Player {target} ({max_vote}표)"
                        )
                        summary_lines.append(
                            f" -> Player {target}를 처형하시겠습니까? (팀 승리를 위해 판단하세요)"
                        )
                    else:
                        summary_lines.append(
                            f"동점자가 발생하여 처형이 진행되지 않을 예정입니다. (후보들: {candidates})"
                        )

        death_summary = []
        for event in status.action_history:
            if (
                event.event_type == EventType.EXECUTE
                and event.target_id != -1
                and event.value is not None
            ):
                role_name = (
                    self.logger.get_role_korean_name(event.value)
                    if self.logger
                    else event.value.name
                )
                death_summary.append(
                    f"{event.day}일차 낮: Player {event.target_id} 처형됨 (직업: {role_name})"
                )
            elif event.event_type == EventType.KILL and event.phase == Phase.NIGHT:
                death_summary.append(
                    f"{event.day}일차 밤: Player {event.target_id} 살해당함"
                )
            elif (
                event.event_type == EventType.SYSTEM_MESSAGE
                and event.phase == Phase.DAY_DISCUSSION
            ):
                if event.target_id != -1:
                    death_summary.append(
                        f"{event.day}일차 아침: 지난 밤 Player {event.target_id} 사망"
                    )
                else:
                    death_summary.append(f"{event.day}일차 아침: 지난 밤 사망자 없음")

        if death_summary:
            summary_lines.append("\n* 사망자 정보:")
            summary_lines.extend([f" - {s}" for s in sorted(list(set(death_summary)))])

        my_night_actions = []
        for event in status.action_history:
            if event.phase == Phase.NIGHT and event.actor_id == status.my_id:
                if (
                    event.event_type == EventType.POLICE_RESULT
                    and event.value is not None
                ):
                    role_name = (
                        self.logger.get_role_korean_name(event.value)
                        if self.logger
                        else event.value.name
                    )
                    my_night_actions.append(
                        f"{event.day}일차 밤: Player {event.target_id}을(를) 조사하여 '{role_name}'임을 확인"
                    )
                elif event.event_type == EventType.PROTECT:
                    my_night_actions.append(
                        f"{event.day}일차 밤: Player {event.target_id}을(를) 보호함"
                    )

        if my_night_actions:
            summary_lines.append("\n* 나의 비공개 행동 결과:")
            summary_lines.extend([f"  - {s}" for s in my_night_actions])

        if not summary_lines:
            return "아직 요약할 정보가 없습니다."

        return "\n".join(summary_lines)
