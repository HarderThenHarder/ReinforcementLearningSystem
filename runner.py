"""
@ Author: Szh
@ Time: 2020/2/25
@ Software: PyCharm
"""
import math
import numpy as np


def polar_coordinates_from_two_points(x: list, y: list):
    # polar coordinates of x with respect to y
    # angle starts at positive half axis of x with (0, 2*pi) in radian
    angle = math.atan2(x[1] - y[1], x[0] - y[0])
    if(angle < 0):
        angle = angle + math.pi * 2  # wrong version but works: angle = angle + math.pi
    distance = math.sqrt(math.pow(x[0] - y[0], 2) + math.pow(x[1] - y[1], 2))
    return [distance, angle]


def preprocess(x: list, y):
    """
        :param x: the observation from env
        :param y: the time step of the episode
        :return: the observation suitable for the algorithm
    """
    # polar coordinates of uav with respect to four radars
    x[2:4] = polar_coordinates_from_two_points(x[0:2], x[2:4])
    x[4:6] = polar_coordinates_from_two_points(x[0:2], x[4:6])
    x[6:8] = polar_coordinates_from_two_points(x[0:2], x[6:8])
    x[8:10] = polar_coordinates_from_two_points(x[0:2], x[8:10])
    # polar coordinates of uav with respect to the command
    x[10:12] = polar_coordinates_from_two_points(x[0:2], x[10:12])
    # the marching direction of uav(direction of cartesian coordinates (1, 0) in the map as 0 & (-pi, pi))
    x[12] = math.atan2(x[13], x[12])
    # the time flag
    x[13] = y
    return x[2:]


def normalize(x: list):
    """
        :param x: the preprocessed observation with straight meaning
        :return: the normalized observation between -1 and 1
    """
    max_dis = math.sqrt(800 * 800 + 1400 * 1400)
    # normalize the distance
    x[0] = x[0] / max_dis * 2 - 1
    x[2] = x[2] / max_dis * 2 - 1
    x[4] = x[4] / max_dis * 2 - 1
    x[6] = x[6] / max_dis * 2 - 1
    x[8] = x[8] / max_dis * 2 - 1
    # normalize the angle
    x[1] = x[1] / math.pi - 1
    x[3] = x[3] / math.pi - 1
    x[5] = x[5] / math.pi - 1
    x[7] = x[7] / math.pi - 1
    x[9] = x[9] / math.pi - 1
    # normalize the uav direction and time flag
    x[10] = x[10] / math.pi
    x[11] = x[11] / 1000
    return x


def shaping_reward(x: list, y: bool, z: bool):
    """
        :param x: the preprocessed observation with straight meaning
        :param y: flag indicate whether the uav has marched into the radar area
        :param z: flag indicate whether the uav has marched out of the radar area
        :return: the shaped reward according to the observation and the updated flag
    """
    if(not y):
        if(x[0] < 200 or x[2] < 200 or x[4] < 200 or x[6] < 200):
            # marching into the radar area
            y = True
            r = 1
        else:
            # beginning position outside
            r = 0
    elif(y and not z):
        if(x[0] > 200 and x[2] > 200 and x[4] > 200 and x[6] > 200):
            if(x[1] < 0.75 * math.pi):
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
        if(x[8] < 100):
            # the command got
            r = 1
        else:
            # searching the command
            r = 0
    return r, y, z


class Runner(object):
    def __init__(self, env, model, nsteps, lam, gamma, render):
        self.env = env
        self.model = model
        self.nsteps = nsteps  # how many steps needed for an interaction loop
        self.lam = lam  # Lambda used in GAE (General Advantage Estimation)
        self.gamma = gamma  # Discount rate
        self.render = render  # indicate whether to render the environment
        self.step = 0  # record the steps within an episode
        self.episode = 1  # record the current episode number
        self.total_step = 0  # record the total steps for which the model has interacted with the env
        self.total_reward = 0.0  # record the cumulative rewards within an episode
        self.in_radar = False  # indicate if the uav has marched into the radar area
        self.out_radar = False  # indicate if the uav has marched out of the radar area

        # initialize the observation as the reset environment and done flag as True
        # batch of env considered, though not implemented
        self.obs = np.zeros((1, model.obs_dim), dtype=np.float32)
        self.obs[:] = np.expand_dims(np.array(normalize(preprocess(self.env.reset(), self.step))), axis=0)
        self.done = np.asarray([True], dtype=np.bool)

    def run(self):
        # initialize the lists that will contain the batch of experiences and the stats of finished episodes
        b_obs, b_rew, b_act, b_opv, b_olp, b_done, t_rew_list, t_len_list= [], [], [], [], [], [], [], []

        for _ in range(self.nsteps):
            # Given observations, get action, value and logp
            if(self.render):
                self.env.render()
            act, opv, olp = self.model.step(self.obs)
            b_obs.append(self.obs.copy())
            b_act.append(act)
            b_opv.append(opv)
            b_olp.append(olp)

            red_action_list = [[act[0]], [1, 1, 1]]  # [[Attack UAV1], [Jamming UAV1, Jamming UAV2, Jamming UAV3]]
            blue_action_list = []
            obs, r, done = self.env.step(red_action_list, blue_action_list)
            self.step += 1
            self.total_step += 1
            obs = preprocess(obs, self.step)
            rew, self.in_radar, self.out_radar = shaping_reward(obs, self.in_radar, self.out_radar)
            self.total_reward += rew
            if done:
                if(r != 500):
                    rew = -1
                    self.total_reward += rew
                t_len_list.append(self.step)
                t_rew_list.append(self.total_reward)
                print('Total step {}: Total cumulative reward of episode {} with step {} is {}.'.format(
                    self.total_step, self.episode, self.step, self.total_reward))
                self.step = 0
                self.episode += 1
                self.total_reward = 0.0
                self.in_radar = False
                self.out_radar = False
                self.obs[:] = np.expand_dims(np.array(normalize(preprocess(self.env.reset(), self.step))), axis=0)
            else:
                self.obs[:] = np.expand_dims(np.array(normalize(obs)), axis=0)
            self.done = np.asarray([done], dtype=np.bool)
            b_rew.append([rew])
            b_done.append(self.done)

        b_obs = np.asarray(b_obs, dtype=self.obs.dtype)  # observation received
        b_act = np.asarray(b_act, dtype=np.uint8)  # action taken based on the observation
        b_rew = np.asarray(b_rew, dtype=np.float32)  # reward received based on the obs and act
        b_opv = np.asarray(b_opv, dtype=np.float32)  # the predicted value of the observation back then
        b_olp = np.asarray(b_olp, dtype=np.float32)  # the log probability of the action taken
        b_done = np.asarray(b_done, dtype=np.bool)  # if the episode is done after the taken action
        last_value = self.model.value(self.obs)  # calculate the value of last obs in the loop for bootstrapping

        # generalized advantage estimation
        b_adv = np.zeros_like(b_rew)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            nextnonterminal = 1.0 - b_done[t]
            if t == self.nsteps - 1:
                nextvalues = last_value
            else:
                nextvalues = b_opv[t + 1]
            delta = b_rew[t] + self.gamma * nextvalues * nextnonterminal - b_opv[t]
            b_adv[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        b_ret = b_adv + b_opv

        return (*map(sf01, (b_obs, b_ret, b_act, b_opv, b_olp)), t_len_list, t_rew_list)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1(for the case of parallel environment)
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

