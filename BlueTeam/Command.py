"""
@ Author: Pky
@ Time: 2020/1/31
@ Software: PyCharm 
"""
from GameObject import GameObject
import pygame.gfxdraw as gfx


class Command(GameObject):

    def __init__(self, pos:list, img_path: str, index: int):
        super().__init__(pos, img_path, index)
        self.warning_area_r = 100

    def render(self, screen):
        gfx.filled_circle(screen, self.pos[0], self.pos[1], self.warning_area_r, (50, 50, 100, 80))
        screen.blit(self.img, self.draw_pos)

    def update(self, **kwargs) -> bool:
        done = False
        return done