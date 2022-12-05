import random
from abc import *


class BasePlayer(metaclass=ABCMeta):
    def __init__(self, index: int, num_white_dice: int = 0, print_game: bool = True):
        self.index = index
        self._dice = []
        self._dice_white = []
        self._num_white_dice = num_white_dice
        self._money = 0
        self._print_game = print_game
        self.reset_game()

    def __str__(self):
        return "- Player{}\tMoney: {:<8}\tDice: {:<24}{:<6}".format(self.index, self._money, str(self._dice), str(self._dice_white))

    def reset_game(self):
        self._dice = [1 for _ in range(8)]
        self._dice_white = [1 for _ in range(self._num_white_dice)]
        self._money = 0

    def reset_round(self):
        self._dice = [1 for _ in range(8)]
        self._dice_white = [1 for _ in range(self._num_white_dice)]

    def set_num_white_dice(self, num_white_dice: int):
        self._num_white_dice = num_white_dice

    def add_banknotes(self, banknotes: list):
        if isinstance(banknotes, int):
            self._money += banknotes
        elif isinstance(banknotes, list):
            for banknote in banknotes:
                self._money += banknote

    def get_money(self):
        return self._money

    def get_num_dice(self):
        return len(self._dice), len(self._dice_white)

    def _roll_dice(self):
        for i in range(len(self._dice)):
            self._dice[i] = random.randint(1, 6)
        for i in range(len(self._dice_white)):
            self._dice_white[i] = random.randint(1, 6)

        # Sort Dice
        self._dice.sort()
        self._dice_white.sort()

    @abstractmethod
    def _select_casino(self, **kwargs):
        pass

    def run_turn(self, **kwargs):
        output_dice = []

        if len(self._dice) == 0 and len(self._dice_white) == 0:
            return False, []

        self._roll_dice()
        if self._print_game and "game_info" in kwargs.keys():
            print(kwargs["game_info"]['text'])
        select = self._select_casino(**kwargs)

        dice_to_remove = []
        for i, d in enumerate(self._dice):
            if d == select:
                output_dice.append(self.index)
                dice_to_remove.append(i)
        for d_i in sorted(dice_to_remove, reverse=True):
            self._dice.pop(d_i)

        dice_white_to_remove = []
        for i, d in enumerate(self._dice_white):
            if d == select:
                output_dice.append(0)
                dice_white_to_remove.append(i)
        for d_i in sorted(dice_white_to_remove, reverse=True):
            self._dice_white.pop(d_i)

        return select, output_dice
