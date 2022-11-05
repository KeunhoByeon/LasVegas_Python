from .base_player import BasePlayer


class HumanPlayer(BasePlayer):
    def __init__(self, index):
        super(HumanPlayer, self).__init__(index)

    def _select_casino(self):
        while True:
            print("[Player{}]  Dice: {}{}".format(self.index, str(self._dice), str(self._dice_white)))
            select = input('Select Casino: ')
            if not select.isnumeric():
                continue

            if int(select) in self._dice or int(select) in self._dice_white:
                return int(select)
