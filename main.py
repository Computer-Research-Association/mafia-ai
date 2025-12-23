import argparse
from core.env import MafiaEnv
from ai.ppo import PPO # (나중에 구현 예정)
from ai.reinforce import REINFORCEAgent

def main(args):
    # 1. 환경 생성
    log_file_path = "mafia_game_log.txt"
    with open(log_file_path, "w", encoding="utf-8") as f:
        env = MafiaEnv(log_file=f)
        # PPO 대신 REINFORCE 에이전트 생성
        agent = REINFORCEAgent(state_dim=env.observation_space.shape[0],
                               action_dim=env.action_space.n)

        if args.mode == 'train':
            print("Sanity Check Training Started (REINFORCE)...")

            for episode in range(1000):  # 1000판만 돌려보자
                f.write(f"\n{'='*20} 에피소드 {episode} 시작 {'='*20}\n")
                obs, _ = env.reset()
                done = False
                total_reward = 0

                while not done:
                    # 1. 행동 선택
                    action = agent.select_action(obs)

                    # 2. 환경 진행
                    next_obs, reward, done, truncated, _ = env.step(action)

                    # 3. ★중요★ 보상을 에이전트에게 알려줌 (REINFORCE는 이게 필요)
                    agent.rewards.append(reward)

                    obs = next_obs
                    total_reward += reward

                # 4. 한 게임 끝나면 바로 학습 (Update)
                agent.update()

                f.write(f"\n에피소드 {episode} 종료. 총 보상: {total_reward}\n")
                f.write(f"{'='*50}\n")
                if episode % 100 == 0:
                    print(f"Episode {episode}, Total Reward: {total_reward}")

        elif args.mode == 'test':
            print("Test Mode Started...")
            obs, _ = env.reset()
            done = False
            while not done:
                # 임시 랜덤 행동
                action = env.action_space.sample()
                obs, reward, done, _, _ = env.step(action)
                env.render()
            print("Game Over")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'], help='train or test')
    args = parser.parse_args()
    main(args)