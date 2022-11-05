import random

from .base_player import BasePlayer


class RuleBasePlayer(BasePlayer):
    """
    Rule Base Player is not yet implemented.
    This will be selecting a casino randomly.
    """
    def __init__(self, index, random_ratio: float = 1.0):
        super(RuleBasePlayer, self).__init__(index)
        self.random_ratio = random_ratio

    def _select_casino_randomly(self):
        while True:
            select = random.randint(1, 6)
            if int(select) in self._dice or int(select) in self._dice_white:
                break
        return select

    def _select_casino_rule_based(self):
        # TODO: Not yet implemented!
        return self._select_casino_randomly()

    def _select_casino(self):
        print("[Player{}]  Dice: {}{}".format(self.index, str(self._dice), str(self._dice_white)))
        if random.random() < self.random_ratio:
            select = self._select_casino_randomly()
        else:
            select = self._select_casino_rule_based()
        print("Select Casino: {}".format(select))
        return select
