"""
@ Author: Pky
@ Time: 2020/2/2
@ Software: PyCharm 
"""
import json
from RedTeam.RedTeam import RedTeam
from BlueTeam.BlueTeam import BlueTeam
from EMCSimulator import EMCSimulator
from AlgorithmsLib.PolicyGradient import PolicyGradient
from AlgorithmsLib.DQN import DQN
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import shutil


Epoch = 5000
MAXSTEP = 1e5


def PG_brain():
    config = json.load(open("config.json", 'r'))
    red_team_object_list = config["red_team"]
    blue_team_object_list = config["blue_team"]

    red_team = RedTeam(red_team_object_list)
    blue_team = BlueTeam(blue_team_object_list)
    simulator = EMCSimulator(1400, 800, blue_team, red_team)

    PG = PolicyGradient(observation_dim=20, action_dim=3)

    for i_eposide in range(Epoch):
        obs = simulator.reset()
        obs = torch.FloatTensor(obs)

        step = 0
        while True:
            step += 1
            simulator.render()
            action = PG.choose_action(obs)
            red_action_list = [[action], [1, 1, 1]]       # [[Attack UAV1], [Jamming UAV1, Jamming UAV2, Jamming UAV3]]
            blue_action_list = []                         # Not set yet
            obs_, r, done = simulator.step(red_action_list, blue_action_list)
            PG.store_transition(obs, r, action)

            if done or step > MAXSTEP:
                ep_r = sum(PG.ep_r)
                loss = PG.learn()
                print("Episode: %d | Reward: %d | Loss: %.4f" % (i_eposide, ep_r, loss.item()))
                break
            obs = torch.FloatTensor(obs_)


def DQN_brain():
    render_flag = True
    config = json.load(open("config.json", 'r'))
    red_team_object_list = config["red_team"]
    blue_team_object_list = config["blue_team"]

    red_team = RedTeam(red_team_object_list)
    blue_team = BlueTeam(blue_team_object_list)
    simulator = EMCSimulator(1400, 800, blue_team, red_team, render=True)

    dqn = DQN(observation_dim=20, action_dim=3, memory_capacity=1000)

    log_name = "run/DQN_brain_no_render"
    if os.path.exists(log_name):
        shutil.rmtree(log_name)
    writer = SummaryWriter(log_name)
    writer.add_graph(dqn.evaluate_net, torch.randn(1, dqn.observation_dim))

    for i_eposide in range(Epoch):
        obs = simulator.reset()
        obs = torch.FloatTensor(obs)
        running_loss = 0
        cumulative_reward = 0
        step = 0
        while True:
            step += 1
            if render_flag:
                simulator.render()
            action = dqn.choose_action(obs)
            red_action_list = [[action], [1, 1, 1]]       # [[Attack UAV1], [Jamming UAV1, Jamming UAV2, Jamming UAV3]]
            blue_action_list = []                         # Not set yet
            obs_, r, done = simulator.step(red_action_list, blue_action_list)
            dqn.store_transition(obs, action, r, obs_)

            # It means uav has arrived at enemy's command
            if r == 500:
                # render_flag = True
                torch.save(dqn.evaluate_net, "./models/dqn/evaluate_net_%d.pth" % i_eposide)
                torch.save(dqn.target_net, "./models/dqn/target_net_%d.pth" % i_eposide)
                print("\n -> Model has saved at: './models/dqn/xx.pth'\n")


            cumulative_reward += r
            if dqn.point > dqn.memory_capacity:
                loss = dqn.learn()
                running_loss += loss
                if done or step > MAXSTEP:
                    writer.add_scalar("training/Loss", running_loss / step, dqn.learn_step)
                    writer.add_scalar("training/Reward", cumulative_reward, dqn.learn_step)
                    writer.add_scalar("training/Exploration", dqn.epsilon, dqn.learn_step)
                    print("\n - Episode: %d Cumulative Reward: %.2f\n" % (i_eposide, cumulative_reward))
                    break
                elif step % 100 == 99:
                    print("Episode: %d| Global Step: %d| Loss:  %.4f, Reward: %.2f, Exploration: %.4f" % (i_eposide, dqn.learn_step, running_loss / step, cumulative_reward, dqn.epsilon))
            else:
                print("\rCollecting experience: %d / %d..." % (dqn.point, dqn.memory_capacity), end='')

            if done:
                break
            obs = torch.FloatTensor(obs_)

if __name__ == '__main__':
    # PG_brain()
    DQN_brain()
