import random

from .base_player import BasePlayer


class RandomPlayer(BasePlayer):
    def __init__(self, index, print_game: bool = True):
        super(RandomPlayer, self).__init__(index, print_game=print_game)

    def _select_casino_randomly(self):
        while True:
            select = random.randint(1, 6)
            if int(select) in self._dice or int(select) in self._dice_white:
                break
        return select

    def _select_casino(self, **kwargs):
        print("[Player{}]  Dice: {}{}".format(self.index, str(self._dice), str(self._dice_white))) if self._print_game else None
        select = self._select_casino_randomly()
        print("Select Casino: {}".format(select)) if self._print_game else None
        return select
