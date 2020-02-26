"""
@ Author: Pky
@ Time: 2020/1/31
@ Software: PyCharm 
"""
from .Radar import Radar
from .AntiUAVCar import AntiUAVCar
from .Command import Command
from .Fort import Fort
from Team import Team


class BlueTeam(Team):

    def __init__(self, team_object_dict: dict):
        super().__init__()
        self.object_dict = {"radar": [], "fort": [], "antiuavcar": [], "command": []}
        self.team_object_dict = team_object_dict
        self.build_team()

    def build_team(self):
        assert "radar" in self.team_object_dict, "Blue Team Must have 'radar' object, check if there exists 'radar' object in 'team_object_dict'."
        for radar in self.team_object_dict["radar"]:
            self.object_dict["radar"].append(
                Radar(radar["pos"], radar["img_path"], radar["index"], radar["detect_r"], radar["kernel_r"], radar["rotate_speed"]))

        assert "fort" in self.team_object_dict, "Blue Team Must have 'fort' object, check if there exists 'fort' object in 'team_object_dict'."
        for fort in self.team_object_dict["fort"]:
            self.object_dict["fort"].append(Fort(fort["pos"], fort["img_path"], fort["index"]))

        assert "antiuavcar" in self.team_object_dict, "Blue Team Must have 'antiuavcar' object, check if there exists 'antiuavcar' object in 'team_object_dict'."
        for car in self.team_object_dict["antiuavcar"]:
            self.object_dict["antiuavcar"].append(AntiUAVCar(car["pos"], car["img_path"], car["index"], car["velocity"]))

        assert "command" in self.team_object_dict, "Blue Team Must have 'command' object, check if there exists 'command' object in 'team_object_dict'."
        for command in self.team_object_dict["command"]:
            self.object_dict["command"].append(Command(command["pos"], command["img_path"], command["index"]))

    def reset(self):
        self.object_dict = {"radar": [], "fort": [], "antiuavcar": [], "command": []}
        self.build_team()
