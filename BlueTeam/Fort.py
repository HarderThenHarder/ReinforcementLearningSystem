"""
@ Author: Pky
@ Time: 2020/1/31
@ Software: PyCharm 
"""
from GameObject import GameObject


class Fort(GameObject):

    def __init__(self, pos:list, img_path: str, index: int):
        super().__init__(pos, img_path, index)

    def update(self, **kwargs) -> bool:
        done = False
        return done