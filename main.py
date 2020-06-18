"""
@Author: P_k_y
@Time: 2020/6/13
"""

import json
from RedTeam.RedTeam import RedTeam
from BlueTeam.BlueTeam import BlueTeam
from EMCSimulator import EMCSimulator
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np
from AlgorithmsLib.PPO_torch import PPO
import math

Epoch = 500000
MAXSTEP = 1e5
UPDATE_INTERVAL = 5000
RENDER = False
LOGGING_INTERVAL = 20


def polar_coordinates_from_two_points(x: list, y: list):
    # polar coordinates of x with respect to y
    # angle starts at positive half axis of x with (0, 2*pi) in radian
    angle = math.atan2(x[1] - y[1], x[0] - y[0])
    if (angle < 0):
        angle = angle + math.pi * 2  # wrong version but works: angle = angle + math.pi
    distance = math.sqrt(math.pow(x[0] - y[0], 2) + math.pow(x[1] - y[1], 2))
    return [distance, angle]


def normalize(x: np.array):
    """
        :param x: the preprocessed observation with straight meaning
        :return: the normalized observation between -1 and 1
    """
    max_dis = math.sqrt(800 * 800 + 1400 * 1400)

    # normalize the distance
    x[0] = x[0] / max_dis * 2 - 1
    x[3] = x[3] / max_dis * 2 - 1
    x[6] = x[6] / max_dis * 2 - 1
    x[9] = x[9] / max_dis * 2 - 1
    x[12] = x[12] / max_dis * 2 - 1

    # normalize the angle
    x[1] = x[1] / math.pi - 1
    x[2] = x[2] / math.pi - 1
    x[4] = x[4] / math.pi - 1
    x[5] = x[5] / math.pi - 1
    x[7] = x[7] / math.pi - 1
    x[8] = x[8] / math.pi - 1
    x[10] = x[10] / math.pi - 1
    x[11] = x[11] / math.pi - 1
    x[13] = x[13] / math.pi - 1
    x[14] = x[14] / math.pi - 1

    return x


def preprocess_obs(x: list):
    """
        :param x: the observation from env
        :param y: the time step of the episode
        :return: the observation suitable for the algorithm
    """
    # # polar coordinates of uav with respect to four radars
    # x[2:4] = polar_coordinates_from_two_points(x[0:2], x[2:4])
    # x[4:6] = polar_coordinates_from_two_points(x[0:2], x[4:6])
    # x[6:8] = polar_coordinates_from_two_points(x[0:2], x[6:8])
    # x[8:10] = polar_coordinates_from_two_points(x[0:2], x[8:10])
    # # polar coordinates of uav with respect to the command
    # x[10:12] = polar_coordinates_from_two_points(x[0:2], x[10:12])
    # # the marching direction of uav(direction of cartesian coordinates (1, 0) in the map as 0 & (-pi, pi))
    # x[12] = math.atan2(x[13], x[12])
    # # the time flag
    # x[13] = y

    # 处理观测向量
    x[2:4] = polar_coordinates_from_two_points(x[0:2], x[2:4])
    x[4] = math.atan2(x[5], x[4])
    x[5:7] = polar_coordinates_from_two_points(x[0:2], x[6:8])
    x[7] = math.atan2(x[9], x[8])
    x[8:10] = polar_coordinates_from_two_points(x[0:2], x[10:12])
    x[10] = math.atan2(x[13], x[12])
    x[11:13] = polar_coordinates_from_two_points(x[0:2], x[14:16])
    x[13] = math.atan2(x[17], x[16])
    x[14:16] = polar_coordinates_from_two_points(x[0:2], x[18:20])
    x[16] = math.atan2(x[21], x[20])

    return normalize(np.array(x[2:17]))       # 15-dim obs


def shaping_reward(x: list, y: bool, z: bool):
    """
        :param x: the preprocessed observation with straight meaning
        :param y: flag indicate whether the uav has marched into the radar area
        :param z: flag indicate whether the uav has marched out of the radar area
        :return: the shaped reward according to the observation and the updated flag
    """
    max_dis = math.sqrt(800 * 800 + 1400 * 1400)
    normalize_radar_detect_r = 200 / max_dis * 2 - 1
    normalize_command_detect_r = 100 / max_dis * 2 - 1

    if not y:
        if x[0] < normalize_radar_detect_r or x[3] < normalize_radar_detect_r or x[6] < normalize_radar_detect_r or x[9] < normalize_radar_detect_r:
            # marching into the radar area
            y = True
            r = 1
        else:
            # beginning position outside
            r = 0
    elif y and not z:
        if x[0] > normalize_radar_detect_r and x[3] > normalize_radar_detect_r and x[6] > normalize_radar_detect_r and x[9] > normalize_radar_detect_r:
            if x[12] < 630 / max_dis * 2 - 1:
                # marching to command out of the radar area
                r = 1
                z = True
            else:
                # marching back out of the radar area
                r = -1
                y = False
        else:
            # marching in the radar area
            r = 0
    else:
        # radar area passed
        if x[12] < normalize_command_detect_r:
            # the command got
            r = 1
        else:
            # searching the command
            r = 0
    return r, y, z


def trainPPO():
    config = json.load(open("config.json", 'r'))
    red_team_object_list = config["red_team"]
    blue_team_object_list = config["blue_team"]

    red_team = RedTeam(red_team_object_list)
    blue_team = BlueTeam(blue_team_object_list)
    simulator = EMCSimulator(1400, 800, blue_team, red_team)

    ppo = PPO(state_dim=15, action_dim=3)

    log_name = "run/PPO_tensorflow"
    if os.path.exists(log_name):
        shutil.rmtree(log_name)
    writer = SummaryWriter(log_name)

    running_reward = 0
    time_step = 0
    total_time_step = 0

    for i_episode in range(Epoch):
        in_radar_flag = out_radar_flag = False
        obs = simulator.reset()
        obs = preprocess_obs(obs)

        for t in range(int(MAXSTEP)):
            if RENDER:
                simulator.render()
            total_time_step += 1
            time_step += 1
            action = ppo.policy_old.act(obs, ppo.memory)
            red_action_list = [[action], [1, 1, 1]]  # [[Attack UAV1], [Jamming UAV1, Jamming UAV2, Jamming UAV3]]
            blue_action_list = []  # Not set yet
            obs_, r, done = simulator.step(red_action_list, blue_action_list)
            obs = preprocess_obs(obs_)

            reward, in_radar_flag, out_radar_flag = shaping_reward(obs, in_radar_flag, out_radar_flag)
            if done and r != 1:  reward -= 1

            ppo.memory.rewards.append(reward)
            ppo.memory.is_terminals.append(done)

            if time_step % UPDATE_INTERVAL == 0:
                ppo.update()
                time_step = 0

            running_reward += reward
            if done:
                break

        if i_episode % LOGGING_INTERVAL == 0:
            running_reward = float("%.2f" % (running_reward / LOGGING_INTERVAL))
            print('Episode {} \t|\t reward: {}'.format(i_episode, running_reward))
            writer.add_scalar("training/Reward", running_reward, i_episode)
            if running_reward == 3:
                torch.save(ppo.policy_old, "./models/PPO_tensorflow/ppo_%d.pth" % i_episode)
            running_reward = 0


def testPPO():
    config = json.load(open("config.json", 'r'))
    red_team_object_list = config["red_team"]
    blue_team_object_list = config["blue_team"]

    red_team = RedTeam(red_team_object_list)
    blue_team = BlueTeam(blue_team_object_list)
    simulator = EMCSimulator(1400, 800, blue_team, red_team)

    ppo = PPO(state_dim=15, action_dim=3)
    ppo.policy_old = torch.load("./models/PPO/ppo_17680.pth")

    with torch.no_grad():
        while True:
            obs = simulator.reset()
            obs = preprocess_obs(obs)
            running_reward = 0
            for i in range(2000):
                simulator.render()

                action = ppo.policy_old.act(obs, ppo.memory)
                red_action_list = [[action], [1, 1, 1]]  # [[Attack UAV1], [Jamming UAV1, Jamming UAV2, Jamming UAV3]]
                blue_action_list = []  # Not set yet
                obs_, r, done = simulator.step(red_action_list, blue_action_list)
                obs = preprocess_obs(obs_)

                ppo.memory.clear_memory()
                running_reward += r

                if done:
                    break


if __name__ == '__main__':
    # trainPPO()
    testPPO()