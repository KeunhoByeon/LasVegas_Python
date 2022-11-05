from .base_player import BasePlayer


class EmptyPlayer(BasePlayer):
    def __init__(self, index):
        super(EmptyPlayer, self).__init__(index)
        self._dice = []
        self._dice_white = []
        self._num_white_dice = 0
        self._money = 0

    def __str__(self):
        return "- Empty"

    def reset_game(self):
        pass

    def reset_round(self):
        pass

    def _select_casino(self):
        pass
