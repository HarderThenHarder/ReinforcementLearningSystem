"""
@ Author: Pky
@ Time: 2020/1/31
@ Software: PyCharm 
"""
import math
import numpy as np


class MathUtils(object):

    @staticmethod
    def rotate(coordinate: list, angle: float) -> list:
        radians = math.radians(angle)
        x = coordinate[0] * math.cos(radians) - math.sin(radians) * coordinate[1]
        y = coordinate[1] * math.cos(radians) + math.sin(radians) * coordinate[0]
        return [x, y]

    @staticmethod
    def get_angle_from_two_points(point1: list, point2: list):
        return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))

    @staticmethod
    def len(vector: list):
        return math.hypot(vector[0], vector[1])

    @staticmethod
    def normalize(vector: list):
        length = MathUtils.len(vector)
        return [vector[0] / length, vector[1] / length]

    @staticmethod
    def get_angle_from_two_vectors(vector1: list, vector2: list):
        x = np.array(MathUtils.normalize(vector1))
        y = np.array(MathUtils.normalize(vector2))
        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))
        frac_up = x.dot(y)
        frac_down = Lx * Ly
        if abs(frac_up - frac_down) < 1e-5:
            return 0
        else:
            return math.degrees(np.arccos(x.dot(y) / (Lx * Ly)))

    @staticmethod
    def get_distance(point1: list, point2: list):
        return math.hypot(point1[0] - point2[0], point1[1] - point2[1])
