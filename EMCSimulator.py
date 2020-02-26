"""
@ Author: Pky
@ Time: 2020/1/31
@ Software: PyCharm 
"""
import pygame
from pygame.locals import *
from RedTeam.RedTeam import RedTeam
from BlueTeam.BlueTeam import BlueTeam
import json
import time
from utilis.MathUtils import MathUtils


TIME_SCALE_LIST = [1, 50, 100, 200]     # speed up the simulation
TIME_SCALE_INDEX = 3

class EMCSimulator(object):

    def __init__(self, width, height, blue_team, red_team, render=True):
        self.width = width
        self.height = height
        if render: self.init_pygame()
        self.render_flag = render
        self.red_team = red_team
        self.blue_team = blue_team
        self.reward = 0
        self.last_time = time.clock()

    def write_env_info(self, text_list, screen, my_font):
        for i, text in enumerate(text_list):
            screen.blit(my_font.render(text, True, (0, 0, 0)), (10, 10 + i * 20))

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), 0, 32)
        pygame.display.set_caption("EMC Simulation -v1.0 Made by Pky @FPS: 0")
        self.bg = pygame.image.load("img/bg.png")
        self.bg = pygame.transform.scale(self.bg, (self.width, self.height))
        self.my_font = pygame.font.SysFont("timesnewroman", 16)


    def reset(self):
        # print("Jump into reset function!")
        self.reward = 0
        self.blue_team.reset()
        self.red_team.reset()
        uav = self.red_team.object_dict["attackuav"][0]
        obs_list = [uav.pos[0], uav.pos[1]]
        for radar in self.blue_team.object_dict["radar"]:
            obs_list.append(radar.pos[0])
            obs_list.append(radar.pos[1])
            obs_list.append(radar.detect_direction[0])
            obs_list.append(radar.detect_direction[1])
        command = self.blue_team.object_dict["command"][0]
        obs_list.append(command.pos[0])
        obs_list.append(command.pos[1])
        return obs_list

    def render(self):
        assert self.render_flag, "Since Render flag is False, you can't use this function."
        global TIME_SCALE_INDEX
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    TIME_SCALE_INDEX = (TIME_SCALE_INDEX + 1) % 4
        self.screen.blit(self.bg, (0, 0))
        self.blue_team.render(self.screen)
        self.red_team.render(self.screen)

        self.write_env_info(["Time Scale: %3d" % TIME_SCALE_LIST[TIME_SCALE_INDEX],
                             "Red Team  : Attack",
                             "Blue Team : Defend "], self.screen, self.my_font)

        pygame.display.update()

    def get_next_state(self):
        uav = self.red_team.object_dict["attackuav"][0]
        obs_list = [uav.pos[0], uav.pos[1]]
        for radar in self.blue_team.object_dict["radar"]:
            obs_list.append(radar.pos[0])
            obs_list.append(radar.pos[1])
            obs_list.append(radar.detect_direction[0])
            obs_list.append(radar.detect_direction[1])
        command = self.blue_team.object_dict["command"][0]
        obs_list.append(command.pos[0])
        obs_list.append(command.pos[1])
        return obs_list

    def check_if_done(self):
        """
        Check if done and calculate the reward.
        :return: bool
        """
        radar_list = self.blue_team.object_dict["radar"]
        attackuav_list = self.red_team.object_dict["attackuav"]

        # calculate the reward
        for radar in radar_list:
            for uav in attackuav_list:

                # check if the uav is out of the map, reward = -200
                if uav.pos[0] < 0 or uav.pos[0] > self.width or uav.pos[1] < 0 or uav.pos[1] > self.height:
                    # print("【System Info】Attack UAV-%d is out of map!" % uav.index)
                    self.reward = -50
                    return True

                # check if uav is be detected, reward = -100
                distance = MathUtils.get_distance(radar.pos, uav.pos)
                if distance < radar.detect_r:
                    if distance < radar.kernel_r:
                        # print("【System Info】Attack UAV-%d is in Radar's kernel area!" % uav.index)
                        self.reward = -100
                        return True
                    else:
                        relative_vec = [uav.pos[0] - radar.pos[0], uav.pos[1] - radar.pos[1]]
                        angle = MathUtils.get_angle_from_two_vectors(relative_vec, radar.detect_direction)
                        if angle < 1:     # if the angle between vector1 (uav to radar) and vector2 (radar's detect direction) < 1 means uav has been detected
                            # print("【System Info】Attack UAV-%d has been Detected!" % uav.index)
                            self.reward = -100
                            return True

                target = self.blue_team.object_dict["command"][0]
                distance = MathUtils.get_distance(target.pos, uav.pos)
                if distance < target.warning_area_r:
                    self.reward = 500
                    return True

                # set the positive reward by the 1 / distance (to target)
                d = MathUtils.get_distance(uav.pos, self.blue_team.object_dict["command"][0].pos)
                self.reward = 1 / d * 500
        return False

    def step(self, red_action_list, blue_action_list):
        # now = time.clock()
        # time_step = now - self.last_time
        # self.last_time = now
        time_step = 0.03
        if time_step > 1e-5:
            pygame.display.set_caption("EMC Simulation -v1.0 Made by Pky @FPS: %d" % (1 / time_step))
        time_step = time_step * TIME_SCALE_LIST[TIME_SCALE_INDEX]
        self.blue_team.update(time_step, self, blue_action_list)
        self.red_team.update(time_step, self, red_action_list)
        s_ = self.get_next_state()
        done = self.check_if_done()

        return s_, self.reward, done


# Test Function
if __name__ == '__main__':
    config = json.load(open("config.json", 'r'))
    red_team_object_list = config["red_team"]
    blue_team_object_list = config["blue_team"]

    red_team = RedTeam(red_team_object_list)
    blue_team = BlueTeam(blue_team_object_list)

    simulator = EMCSimulator(1400, 800, blue_team, red_team)

    while True:
        simulator.render()

        red_action_list = [[1], [1, 1, 1]]       # [[Attack UAV1, Attack UAV2], [Jamming UAV1, Jamming UAV2, Jamming UAV3]]
        blue_action_list = []                   # Not set yet
        s_, r, done = simulator.step(red_action_list, blue_action_list)
        print("State: ", s_)
        print("Reward: ", r)

        if done:
            simulator.reset()
