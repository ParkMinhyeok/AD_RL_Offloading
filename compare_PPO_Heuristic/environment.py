import heapq
import random
import itertools
from collections import deque
from dataclasses import dataclass

from data_loader import load_times_from_txt
from time_models import calculate_client_processing_time, calculate_server_processing_time
import config

@dataclass
class Task:
    task_id: int
    arrival_time: float
    split_point: int = -1
    # 통계 측정을 위한 변수
    completion_time: float = 0.0

class EdgeSimulator:
    def __init__(self):
        # 설정 파일(config.py)에서 파라미터 로드
        self.client_times = load_times_from_txt(config.CLIENT_TIMES_PATH)
        self.server_times = load_times_from_txt(config.SERVER_TIMES_PATH)
        self.simulation_time_limit = config.SIMULATION_TIME

        # 성능 지표
        self.completed_tasks_info = []

    def _initialize_episode_variables(self):
        """에피소드 시작 시 변수 초기화"""
        self.virtual_clock = 0.0
        self.server_busy_until = 0.0
        self.task_queue = deque()
        self.event_queue = []
        self.task_id_counter = itertools.count()
        self.event_id_counter = itertools.count()
        self.completed_tasks_in_step = 0
        self.completed_tasks_info = []

    def reset(self):
        """환경 초기화 (RL용)"""
        self._initialize_episode_variables()
        self._schedule_task_arrival()
        state, _, _, _ = self._run_until_next_decision()
        return state

    def step(self, action: int):
        """액션을 받아 한 스텝 진행 (RL용)"""
        if self.task_queue:
            task_to_process = self.task_queue.popleft()
            task_to_process.split_point = config.ACTION_MAP.get(action, -1)
            self._assign_task_to_server(task_to_process)
        return self._run_until_next_decision()

    def get_heuristic_action(self):
        """휴리스틱 규칙에 따라 최적의 행동을 결정 (Heuristic용)"""
        if self.virtual_clock < self.server_busy_until:
            wait_time = self.server_busy_until - self.virtual_clock
            
            cumulative_time = 0.0
            best_split_point = -1 # 기본값: 전체 오프로딩
            # 서버 대기시간 동안 클라이언트에서 처리 가능한 가장 늦은 분할 지점 찾기
            for i, time_needed in enumerate(self.client_times):
                if cumulative_time + time_needed <= wait_time:
                    cumulative_time += time_needed
                    best_split_point = i
                else:
                    break
            return best_split_point
        else:
            # 서버가 유휴 상태이면 모든 작업을 서버로 보냄
            return -1

    def _run_until_next_decision(self):
        """다음 결정 시점까지 시뮬레이션 진행"""
        self.completed_tasks_in_step = 0
        while self.virtual_clock < self.simulation_time_limit:
            if self.task_queue and self.virtual_clock >= self.server_busy_until:
                state = self._get_state()
                reward = float(self.completed_tasks_in_step)
                done = False
                info = {}
                return state, reward, done, info

            if not self.event_queue: break
            
            event_time, _, event_type, data = heapq.heappop(self.event_queue)
            self.virtual_clock = event_time

            if self.virtual_clock >= self.simulation_time_limit: break

            if event_type == "TASK_ARRIVAL":
                self._handle_task_arrival()
            elif event_type == "SERVER_FINISH":
                self._handle_server_finish(data)
        
        state = self._get_state()
        reward = float(self.completed_tasks_in_step)
        done = True
        info = {'completed_tasks_info': self.completed_tasks_info}
        return state, reward, done, info

    def _get_state(self):
        wait_time = max(0, self.server_busy_until - self.virtual_clock)
        queue_len = len(self.task_queue)
        return [wait_time, queue_len]

    def _schedule_event(self, delay, event_type, data=None):
        event_time = self.virtual_clock + delay
        event_id = next(self.event_id_counter)
        heapq.heappush(self.event_queue, (event_time, event_id, event_type, data))

    def _schedule_task_arrival(self):
        delay = random.uniform(config.TASK_ARRIVAL_MIN, config.TASK_ARRIVAL_MAX)
        self._schedule_event(delay, "TASK_ARRIVAL")

    def _handle_task_arrival(self, data=None):
        task_id = next(self.task_id_counter)
        new_task = Task(task_id, self.virtual_clock)
        self.task_queue.append(new_task)
        self._schedule_task_arrival()

    def _assign_task_to_server(self, task: Task):
        server_processing_start_time = max(self.virtual_clock, self.server_busy_until)
        
        client_time = calculate_client_processing_time(task.split_point, self.client_times)
        server_time = calculate_server_processing_time(task.split_point, self.server_times)

        # 실제 서버 작업 시작 시간 = max(서버 유휴 시작, 클라이언트 처리 완료 시점)
        server_actual_start_time = max(server_processing_start_time, task.arrival_time + client_time)
        task_end_time = server_actual_start_time + server_time
        
        task.completion_time = task_end_time
        self.server_busy_until = task_end_time
        event_id = next(self.event_id_counter)
        heapq.heappush(self.event_queue, (task_end_time, event_id, "SERVER_FINISH", task))

    def _handle_server_finish(self, task: Task):
        self.completed_tasks_in_step += 1
        self.completed_tasks_info.append(task)