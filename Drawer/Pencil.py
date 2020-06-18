"""
@Author: P_k_y
@Time: 2020/6/15
"""

import pygame


class Pencil:

    @staticmethod
    def draw_line(screen, start_pos, end_pos, color=(0, 0, 0), width=1):
        pygame.draw.line(screen, color, start_pos, end_pos, width)

    @staticmethod
    def draw_rect(screen, rect, color=(0, 0, 0), width=0):
        """
        If width != 0, then the rect won't be filled, it will be stroked!

        """
        pygame.draw.rect(screen, color, rect, width)

    @staticmethod
    def draw_alpha_rect(screen, rect, color=(0, 0, 0), alpha=0.8):
        """
        绘制透明矩形框。
        :param screen: screen object
        :param rect: [start_x, start_y, width, height]
        :param color: rect color
        :param alpha: transparency
        :return: None
        """
        alpha_rect = pygame.Surface([rect[2], rect[3]])
        alpha_rect.fill(color)
        alpha_rect.set_alpha(int(alpha * 255))
        screen.blit(alpha_rect, (rect[0], rect[1]))

    @staticmethod
    def draw_poly_rect(screen, color, pointlist, width=0):
        pygame.draw.polygon(screen, color, pointlist, width)

    @staticmethod
    def draw_circle(screen, pos,  radius, color=(0, 0, 0), width=0):
        pygame.draw.circle(screen, color, pos, radius, width)

    @staticmethod
    def draw_arc(screen, rect, start_angle, end_angle, color=(0, 0, 0), width=0):
        pygame.draw.arc(screen, color, rect, start_angle, end_angle, width)

    @staticmethod
    def write_text(screen, content, font_pos: tuple, font_size, color=(0, 0, 0), font_family="omic Sans MS"):
        my_font = pygame.font.SysFont(font_family, font_size)
        text_image = my_font.render(content, False, color)
        screen.blit(text_image, font_pos)

    @staticmethod
    def write_text_list(screen, content_list, start_pos: tuple, line_height=15, font_size=15, color=(0, 0, 0), font_family="omic Sans MS"):
        """
        按照从上往下的方式书写一个列表的文字信息。
        :param line_height: 行间距
        :param screen: screen object
        :param content_list: 文字信息列表。
        :param start_pos: 开始位置
        :param font_size: 字体大小
        :param color: 字体颜色
        :return: None
        """
        for i, content in enumerate(content_list):
            Pencil.write_text(screen, content, (start_pos[0], start_pos[1] + i * line_height), font_size, color, font_family)


