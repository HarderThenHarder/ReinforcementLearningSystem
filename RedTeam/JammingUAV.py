"""
@ Author: Pky
@ Time: 2020/1/31
@ Software: PyCharm 
"""
from GameObject import GameObject
from utilis.MathUtils import MathUtils
import pygame


class JammingUAV(GameObject):

    def __init__(self, pos:list, img_path: str, index: int, velocity: list, action_list: list):
        super().__init__(pos, img_path, index)
        self.velocity = velocity    # velocity of Jamming UAV is 100 m/s -> 360 km/h -> 0.036 km/s
        self.action_list = action_list
        self.rotate_angle = 0
        self.wait_count = 0

    def calculate_draw_pos(self):
        self.draw_pos = (self.pos[0] - self.img.get_size()[0] / 2, self.pos[1] - self.img.get_size()[1] / 2)

    def go(self, time_step):
        self.velocity = MathUtils.rotate(self.velocity, self.rotate_angle)
        self.pos = [self.pos[0] + self.velocity[0] * time_step, self.pos[1] + self.velocity[1] * time_step]
        self.calculate_draw_pos()

    def simple_move(self, time_step):
        self.go(time_step)

    def line_to_target(self, time_step, env):
        target_pos = env.blue_team.object_dict["command"][0].pos
        relative_pos = [target_pos[0] - self.pos[0], target_pos[1] - self.pos[1]]
        self.rotate_angle = MathUtils.get_angle_from_two_vectors(self.velocity, relative_pos)       # y coordinate in window is inverse to numpy
        self.go(time_step)

    def stay(self):
        self.rotate_angle = 0
        self.wait_count += 1
        if self.wait_count == 10:
            self.pos = [self.pos[0], self.pos[1] - 3]
        elif self.wait_count == 20:
            self.pos = [self.pos[0], self.pos[1] + 3]
            self.wait_count = 0
        self.calculate_draw_pos()

    def update(self, **kwargs) -> bool:
        done = False
        time_step = kwargs["time_step"]
        env = kwargs["env"]
        # self.simple_move(time_step)
        self.stay()
        self.img = pygame.transform.rotate(self.img, -self.rotate_angle)
        return done