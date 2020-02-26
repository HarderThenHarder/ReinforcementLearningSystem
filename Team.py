"""
@ Author: Pky
@ Time: 2020/2/1
@ Software: PyCharm 
"""

from abc import abstractmethod, ABCMeta

class Team(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def build_team(self, *args):
        pass

    def render(self, screen):
        assert self.object_dict is not None, "object_list has no elements, make sure you have used 'Team.build_team()' to create the team objects."
        for obj_list in self.object_dict.values():
            for obj in obj_list:
                obj.render(screen)

    def update(self, time_step, env, action_list):
        assert self.object_dict is not None, "object_list has no elements, make sure you have used 'Team.build_team()' to create the team objects."
        for obj_list in self.object_dict.values():
            for obj in obj_list:
                obj.update(time_step=time_step, env=env, action_list=action_list)