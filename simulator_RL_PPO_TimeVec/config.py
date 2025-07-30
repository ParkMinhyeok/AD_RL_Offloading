# --- 실험 환경 설정 ---
SEED = 42
SIMULATION_TIME = 300.0  # 총 시뮬레이션 시간
TASK_ARRIVAL_MIN = 0.1   # 작업 도착 최소 간격
TASK_ARRIVAL_MAX = 0.5   # 작업 도착 최대 간격

# --- 데이터 경로 ---
CLIENT_TIMES_PATH = 'dataset/client_times.txt'
SERVER_TIMES_PATH = 'dataset/server_times(x10).txt'

# --- PPO 에이전트 설정 ---
STATE_SIZE = 109  
ACTION_SIZE = 11
LEARNING_RATE = 0.0001
GAMMA = 0.99
LMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 4
T_HORIZON = 20
NUM_EPISODES = 500

# --- Action Map ---
# RL 에이전트의 action(e.g., 0, 1, 2)을 실제 split_point로 변환
# 여기서는 action이 바로 split_point가 되도록 간단하게 설정
ACTION_MAP = {i: i for i in range(ACTION_SIZE)}