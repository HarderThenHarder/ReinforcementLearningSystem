"""
@ Author: Pky
@ Time: 2020/1/31
@ Software: PyCharm 
"""
from GameObject import GameObject
from utilis.MathUtils import MathUtils
import pygame.gfxdraw as gfx
import math
import pygame
from Drawer.Pencil import Pencil


class Radar(GameObject):

    def __init__(self, pos:list, img_path: str, index: int, detect_r: int, kernel_r: int, rotate_speed: int):
        super().__init__(pos, img_path, index)
        self.detect_r = detect_r
        self.kernel_r = kernel_r
        self.rotate_speed = rotate_speed
        self.start_angle = self.index * 50
        self.detect_direction = [self.detect_r * math.cos(math.radians(self.start_angle)), self.detect_r * math.sin(math.radians(self.start_angle))]

    def render(self, screen):
        gfx.filled_circle(screen, self.pos[0], self.pos[1], self.detect_r, (50, 100, 100, 120))
        gfx.filled_circle(screen, self.pos[0], self.pos[1], self.kernel_r, (200, 50, 50, 120))
        pygame.draw.line(screen, (200, 0, 0), (self.pos[0], self.pos[1]), (int(self.detect_direction[0] + self.pos[0]), int(self.detect_direction[1] + self.pos[1])), 2)
        screen.blit(self.img, self.draw_pos)
        Pencil.write_text(screen, "Radar", (self.draw_pos[0], self.draw_pos[1] + 30), 14, (255, 255, 255),
                          font_family="simsunnsimsun")

    def update(self, **kwargs) -> bool:
        time_step = kwargs["time_step"]
        done = False
        self.detect_direction = MathUtils.rotate(self.detect_direction, self.rotate_speed * time_step)
        return done