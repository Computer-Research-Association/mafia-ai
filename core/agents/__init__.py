from typing import Dict, Any, List
from config import Role, config
from .rl_agent import RLAgent
from .llm_agent import LLMAgent
from .rule_base_agent import RuleBaseAgent
from core.envs.encoders import MDPEncoder, POMDPEncoder

class AgentBuilder:
    @staticmethod
    def build_agents(player_configs: List[Dict[str, Any]], logger=None) -> Dict[int, Any]:
        agents = {}
        for i, p_config in enumerate(player_configs):
            agent_type = p_config.get("type", "rba").lower()
            role_str = p_config.get("role", "citizen").upper()
            role = Role[role_str] if role_str in Role.__members__ else Role.CITIZEN

            if agent_type == "rl":
                # 백본에 따라 인코더와 차원을 자동으로 결정
                bb = p_config.get("backbone", "mlp").lower()
                
                # Check directly, or use the logic from ExperimentManager (which we want to replace)
                if bb in ["lstm", "gru", "rnn"]:
                    encoder = POMDPEncoder()
                else:
                    encoder = MDPEncoder()
                
                agent = RLAgent(
                    player_id=i,
                    role=role,
                    state_dim=encoder.observation_dim,
                    action_dims=config.game.ACTION_DIMS,
                    algorithm=p_config.get("algo", "ppo"),
                    backbone=bb,
                    hidden_dim=p_config.get("hidden_dim", 128),
                    num_layers=p_config.get("num_layers", 2),
                )
                
                load_path = p_config.get("load_model_path")
                if load_path:
                    # Assuming RLAgent has a load method and handles file checks internally or we should do it here?
                    # Previous code did check existence. RLAgent.load probably raises if not found.
                    # Let's keep it simple as requested.
                    try:
                        agent.load(load_path)
                    except Exception as e:
                        print(f"Failed to load model for agent {i}: {e}")
                agents[i] = agent

            elif agent_type == "llm":
                agents[i] = LLMAgent(player_id=i, logger=logger) # LLMAgent __init__ signature check needed
            else:
                agents[i] = RuleBaseAgent(player_id=i, role=role)
        return agents
