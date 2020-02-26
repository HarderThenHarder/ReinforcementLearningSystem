"""
@ Author: Pky
@ Time: 2020/1/31
@ Software: PyCharm 
"""
from .AttackUAV import AttackUAV
from .JammingUAV import JammingUAV
from Team import Team


class RedTeam(Team):

    def __init__(self, team_object_dict: dict):
        super().__init__()
        self.object_dict = {"attackuav": [], "jamminguav": []}
        self.team_object_dict = team_object_dict
        self.build_team()

    def build_team(self):
        assert "attackuav" in self.team_object_dict, "Red Team Must have 'attackuav' object, check if there exists 'attackuav' object in 'team_object_dict'."
        for attackuav in self.team_object_dict["attackuav"]:
            self.object_dict["attackuav"].append(
                AttackUAV(attackuav["pos"], attackuav["img_path"], attackuav["index"], attackuav["velocity"], attackuav["action_list"]))

        assert "jamminguav" in self.team_object_dict, "Red Team Must have 'jamminguav' object, check if there exists 'jamminguav' object in 'team_object_dict'."
        for jamminguav in self.team_object_dict["jamminguav"]:
            self.object_dict["jamminguav"].append(JammingUAV(jamminguav["pos"], jamminguav["img_path"], jamminguav["index"], jamminguav["velocity"], jamminguav["action_list"]))

    def reset(self):
        self.object_dict = {"attackuav": [], "jamminguav": []}
        self.build_team()