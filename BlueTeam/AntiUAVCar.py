"""
@ Author: Pky
@ Time: 2020/1/31
@ Software: PyCharm 
"""
from GameObject import GameObject
import pygame.gfxdraw as gfx


class AntiUAVCar(GameObject):

    def __init__(self, pos:list, img_path: str, index: int, velocity: list):
        super().__init__(pos, img_path, index)
        self.detect_r = 40
        self.velocity = velocity

    def render(self, screen):
        gfx.filled_circle(screen, self.pos[0], self.pos[1], self.detect_r, (100, 50, 100, 50))
        screen.blit(self.img, self.draw_pos)

    def update(self, **kwargs) -> bool:
        done = False
        return done