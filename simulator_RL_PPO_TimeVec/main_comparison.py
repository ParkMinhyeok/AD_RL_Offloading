import torch
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
from torch.distributions import Categorical
import os

import config
from environment import EdgeSimulator
from ppo_agent import PPO_Agent

def set_seed(seed):
    """실험의 재현성을 위해 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(completed_tasks):
    """성능 지표를 계산하는 함수"""
    if not completed_tasks:
        return 0, 0.0
    
    total_completed = len(completed_tasks)
    turnaround_times = [(task.completion_time - task.arrival_time) for task in completed_tasks]
    avg_turnaround_time = sum(turnaround_times) / total_completed if total_completed > 0 else 0
    
    return total_completed, avg_turnaround_time

def run_ppo_experiment(device):
    print("--- PPO 강화학습 실험 시작 ---")
    set_seed(config.SEED)
    
    env = EdgeSimulator()
    agent = PPO_Agent(config.STATE_SIZE, config.ACTION_SIZE, device).to(device)
    
    all_scores = []
    all_actor_losses = []
    all_critic_losses = []

    for e in range(config.NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            for t in range(config.T_HORIZON):
                prob = agent.pi(torch.from_numpy(np.array(state)).float().to(device))
                dist = Categorical(prob)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action).to(device))
                
                next_state, reward, done, _ = env.step(action)
                
                agent.put_data((state, action, reward, next_state, log_prob.item(), done))
                state = next_state
                total_reward += reward
                if done:
                    break
            
            actor_loss, critic_loss = agent.learn()
            all_actor_losses.append(actor_loss)
            all_critic_losses.append(critic_loss)

        all_scores.append(total_reward)

        if e % 10 == 0 and e > 0:
            avg_score = np.mean(all_scores[-10:])
            last_actor_loss = all_actor_losses[-1] if all_actor_losses else -1
            last_critic_loss = all_critic_losses[-1] if all_critic_losses else -1
            print(f"  에피소드 {e}/{config.NUM_EPISODES} | 평균 점수: {avg_score:.2f} | Actor Loss: {last_actor_loss:.4f} | Critic Loss: {last_critic_loss:.4f}")
    
    fig_train, axs_train = plt.subplots(1, 3, figsize=(20, 5))
    fig_train.suptitle('PPO 학습 과정 지표', fontsize=16)

    axs_train[0].plot(all_scores, color='blue')
    axs_train[0].set_title('에피소드별 누적 보상 (점수)')
    axs_train[0].set_xlabel('에피소드')
    axs_train[0].set_ylabel('총 보상')

    axs_train[1].plot(all_actor_losses, color='green')
    axs_train[1].set_title('Actor 손실 함수')
    axs_train[1].set_xlabel('학습 스텝')
    axs_train[1].set_ylabel('손실')

    axs_train[2].plot(all_critic_losses, color='red')
    axs_train[2].set_title('Critic 손실 함수')
    axs_train[2].set_xlabel('학습 스텝')
    axs_train[2].set_ylabel('손실')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\n  학습 완료. 모델을 저장합니다...")
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "ppo_agent_model.pth")
    torch.save(agent.state_dict(), model_path)
    print(f"  모델이 '{model_path}' 경로에 저장되었습니다.")

    print("  학습 완료. 최종 성능 평가 진행...")
    set_seed(config.SEED)
    state = env.reset()
    done = False
    while not done:
        prob = agent.pi(torch.from_numpy(np.array(state)).float().to(device))
        action = torch.argmax(prob).item()
        state, _, done, info = env.step(action)
        
    final_info = info
    return calculate_metrics(final_info.get('completed_tasks_info', []))

def run_heuristic_experiment():
    """휴리스틱 모델 실험을 실행하고 결과를 반환"""
    print("\n--- 휴리스틱 모델 실험 시작 ---")
    set_seed(config.SEED)

    env = EdgeSimulator()
    state = env.reset()
    done = False
    
    while not done:
        action = env.get_heuristic_action()
        state, _, done, info = env.step(action)
    
    final_info = info
    return calculate_metrics(final_info.get('completed_tasks_info', []))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 각 실험 실행
    ppo_completed, ppo_avg_time = run_ppo_experiment(device)
    heuristic_completed, heuristic_avg_time = run_heuristic_experiment()

    # --- 최종 결과 출력 ---
    print("\n\n" + "="*40)
    print("      <<< 최종 실험 결과 비교 >>>")
    print("="*40)
    print(f"\n[PPO 에이전트]")
    print(f"  - 완료된 총 작업 수: {ppo_completed} 개")
    print(f"  - 평균 응답 시간: {ppo_avg_time:.4f} 초")
    
    print(f"\n[휴리스틱 모델]")
    print(f"  - 완료된 총 작업 수: {heuristic_completed} 개")
    print(f"  - 평균 응답 시간: {heuristic_avg_time:.4f} 초")
    print("\n" + "="*40)

    # --- 결과 시각화 ---
    models = ['PPO Agent', 'Heuristic']
    completed_tasks = [ppo_completed, heuristic_completed]
    avg_times = [ppo_avg_time, heuristic_avg_time]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('PPO vs Heuristic 성능 비교', fontsize=16)

    ax[0].bar(models, completed_tasks, color=['royalblue', 'lightcoral'])
    ax[0].set_title('완료된 총 작업 수 (많을수록 좋음)')
    ax[0].set_ylabel('개')
    for i, v in enumerate(completed_tasks):
        ax[0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

    ax[1].bar(models, avg_times, color=['royalblue', 'lightcoral'])
    ax[1].set_title('평균 응답 시간 (적을수록 좋음)')
    ax[1].set_ylabel('초(s)')
    for i, v in enumerate(avg_times):
        ax[1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.rc('font', family='Malgun Gothic')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()