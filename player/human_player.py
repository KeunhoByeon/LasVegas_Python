from .base_player import BasePlayer


class HumanPlayer(BasePlayer):
    def __init__(self, index, print_game: bool = True):
        super(HumanPlayer, self).__init__(index, print_game=print_game)

    def _select_casino(self, **kwargs):
        while True:
            print("[Player{}]  Dice: {}{}".format(self.index, str(self._dice), str(self._dice_white))) if self._print_game else None
            select = input('Select Casino: ')
            if not select.isnumeric():
                continue

            if int(select) in self._dice or int(select) in self._dice_white:
                return int(select)
