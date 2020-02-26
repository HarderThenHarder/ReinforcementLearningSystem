"""
@ Author: Pky
@ Time: 2020/2/2
@ Software: PyCharm 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):

    def __init__(self, observation_dim, action_dim):
        super(Net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.observation_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, self.action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return F.softmax(self.fc5(x))


class PolicyGradient(object):

    def __init__(self, observation_dim, action_dim, learning_rate=0.01, gamma=0.95):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.ep_obs, self.ep_r, self.ep_a = [], [], []
        self.net = Net(observation_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def choose_action(self, observation):
        prob_list = self.net(observation)
        action = np.random.choice(range(prob_list.size(0)), p=prob_list.data.numpy())
        return action

    def store_transition(self, obs, r, a):
        self.ep_obs.append(obs)
        self.ep_r.append(r)
        self.ep_a.append(a)

    def learn(self):
        cumulative_reward_list = self.get_cumulative_reward()
        batch_obs = torch.FloatTensor(np.vstack(self.ep_obs))
        batch_a = torch.LongTensor(np.array(self.ep_a).reshape(-1, 1))
        batch_r = torch.FloatTensor(cumulative_reward_list.reshape(-1, 1))

        action_prob = self.net(batch_obs)
        action_prob.gather(1, batch_a)
        gradient = torch.log(action_prob) * batch_r
        loss = -torch.mean(gradient)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset_epoch_memory()

        return loss.data.numpy()

    def get_cumulative_reward(self):
        running_r = 0
        cumulative_reward = np.zeros_like(self.ep_r)
        for i in reversed(range(len(self.ep_r))):
            running_r = running_r * self.gamma + self.ep_r[i]
            cumulative_reward[i] = running_r

        # normalize cumulative reward
        cumulative_reward -= np.mean(cumulative_reward)
        cumulative_reward /= np.std(cumulative_reward)

        return cumulative_reward

    def reset_epoch_memory(self):
        self.ep_a.clear()
        self.ep_obs.clear()
        self.ep_r.clear()