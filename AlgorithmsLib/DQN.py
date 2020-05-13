"""
@ Author: Pky
@ Time: 2020/2/3
@ Software: PyCharm 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


LR = 0.01                   # learning rate
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency


class Net(nn.Module):

    def __init__(self, observation_dim, action_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(observation_dim, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(20, 10)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(10, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN(object):

    def __init__(self, observation_dim, action_dim, batch_size=32, memory_capacity=1000, exploration_factor=1e-4):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.exploration_factor = exploration_factor
        self.target_net, self.evaluate_net = Net(observation_dim, action_dim), Net(observation_dim, action_dim)
        self.memory = np.zeros((memory_capacity, observation_dim * 2 + 2))      # s, a, r, s_
        self.loss_Function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=LR)
        self.point = 0
        self.learn_step = 0
        self.epsilon = 0.2        # exploration rate

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        self.epsilon = math.exp(-self.learn_step * self.exploration_factor)    # exploration rate
        if np.random.uniform() < (1 - self.epsilon):
            return torch.max(self.evaluate_net(s), 1)[1].data.numpy()[0]
        else:
            return np.random.randint(0, self.action_dim)

    def store_transition(self, s, a, r, s_):
        self.memory[self.point % self.memory_capacity, :] = np.hstack((s, [a, r], s_))
        self.point += 1

    def sample_batch_data(self, batch_size):
        perm_idx = np.random.choice(len(self.memory), batch_size)
        return self.memory[perm_idx, :]

    def learn(self) -> float:
        if self.learn_step % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        self.learn_step += 1

        batch_memory = self.sample_batch_data(self.batch_size)
        batch_state = torch.FloatTensor(batch_memory[:, :self.observation_dim])
        batch_action = torch.LongTensor(batch_memory[:, self.observation_dim: self.observation_dim + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.observation_dim + 1: self.observation_dim + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.observation_dim:])

        q_eval = self.evaluate_net(batch_state)
        q_eval = q_eval.gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_Function(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.numpy()