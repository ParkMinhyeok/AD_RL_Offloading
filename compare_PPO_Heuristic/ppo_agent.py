# ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import config

class PPO_Agent(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(PPO_Agent, self).__init__()
        self.device = device
        self.data = []

        self.fc1 = nn.Linear(state_size, 256)
        self.fc_actor = nn.Linear(256, action_size)
        self.fc_critic = nn.Linear(256, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=config.LEARNING_RATE)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_actor(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_critic(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        s, a, r, s_prime, done_mask, prob_a = \
            torch.tensor(s_lst, dtype=torch.float).to(self.device), \
            torch.tensor(a_lst).to(self.device), \
            torch.tensor(r_lst).to(self.device), \
            torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
            torch.tensor(done_lst, dtype=torch.float).to(self.device), \
            torch.tensor(prob_a_lst).to(self.device)
        
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def learn(self):
        s, a, r, s_prime, done_mask, old_log_prob = self.make_batch()

        for i in range(config.K_EPOCHS):
            td_target = r + config.GAMMA * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = config.GAMMA * config.LMBDA * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            pi = self.pi(s, softmax_dim=1)
            dist = Categorical(pi)
            new_log_prob = dist.log_prob(a.squeeze())

            ratio = torch.exp(new_log_prob - old_log_prob.squeeze()).unsqueeze(1)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.smooth_l1_loss(self.v(s), td_target.detach())

            loss = actor_loss + critic_loss
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()