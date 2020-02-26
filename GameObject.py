"""
@ Author: Pky
@ Time: 2020/1/31
@ Software: PyCharm 
"""
import pygame
from abc import ABCMeta, abstractmethod


class GameObject(metaclass=ABCMeta):

    def __init__(self, pos:list, img_path: str, index: int):
        self.img = pygame.image.load(img_path)
        self.img = pygame.transform.scale(self.img, (25, 25))
        self.pos = pos
        self.draw_pos = (self.pos[0] - self.img.get_size()[0] / 2, self.pos[1] - self.img.get_size()[1] / 2)
        self.index = index

    def render(self, screen):
        screen.blit(self.img, self.draw_pos)

    @abstractmethod
    def update(self, **kwargs):
        pass