import numpy as np

from banknotes_manager import BanknotesManager


class Casino:
    def __init__(self, index, print_game: bool = True):
        self.index = index
        self._banknotes_bucket = []
        self._dice_bucket = []
        self._print_game = print_game

    def __str__(self):
        return "- Casino{}\tBanknotes: {:<21}\tDice: {}".format(self.index, str(self._banknotes_bucket), str(self._dice_bucket))

    def reset_game(self):
        self._banknotes_bucket = []
        self._dice_bucket = []

    def reset_round(self):
        self._banknotes_bucket = []
        self._dice_bucket = []

    def add_banknote(self, price: int):
        self._banknotes_bucket.append(price)
        self._banknotes_bucket.sort(reverse=True)

    def is_over_banknote(self) -> bool:
        return False if np.sum(self._banknotes_bucket) < 50000 else True

    def add_die(self, die_index: int):
        self._dice_bucket.append(die_index)
        self._dice_bucket.sort()

    def add_dice(self, dice: list):
        self._dice_bucket.extend(dice)
        self._dice_bucket.sort()

    def get_dice_order(self) -> list:
        temp_dice_unique, temp_dice_counts = np.unique(self._dice_bucket, return_counts=True)

        counts_to_remove = []
        for i, c1 in enumerate(temp_dice_counts):
            for j, c2 in enumerate(temp_dice_counts):
                if c1 == c2 and i != j and c2 not in counts_to_remove:
                    counts_to_remove.append(c2)

        dice_unique, dice_counts = [], []
        for du, dc in zip(temp_dice_unique, temp_dice_counts):
            if dc not in counts_to_remove:
                dice_unique.append(du)
                dice_counts.append(dc)

        dice_order = np.array(dice_unique)[np.argsort(dice_counts)[::-1]]
        return list(dice_order)

    def pay_out(self) -> dict:
        players_win = {}

        dice_order = self.get_dice_order()
        for banknote in sorted(self._banknotes_bucket, reverse=True):
            if len(dice_order) == 0:
                break
            winner = dice_order.pop(0)
            if winner > 0:
                print("CASINO{} BANKNOTE {}: WINNER IS PLAYER{}!".format(self.index, banknote, winner)) if self._print_game else None
                players_win[winner] = banknote

        return players_win


class CasinosManager:
    def __init__(self, print_game: bool = True):
        self._casinos = [Casino(i + 1, print_game=print_game) for i in range(6)]
        self._print_game = print_game

    def __str__(self):
        return "[CASINOS]\n" + "\n".join((str(caino) for caino in self._casinos))

    def reset_game(self):
        for casino in self._casinos:
            casino.reset_game()

    def reset_round(self):
        for casino in self._casinos:
            casino.reset_round()

    def set_banknotes(self, banknotes_manager: BanknotesManager):
        for casino in self._casinos:
            while not casino.is_over_banknote():
                casino.add_banknote(banknotes_manager.get_banknote())

    def add_dice(self, casino_index: int, dice: list):
        self._casinos[casino_index - 1].add_dice(dice)

    def pay_out(self) -> dict:
        casinos_result = [casino.pay_out() for casino in self._casinos]

        players_win = {}
        for casino_result in casinos_result:
            for player_index, banknote in casino_result.items():
                if player_index not in players_win.keys():
                    players_win[player_index] = []
                players_win[player_index].append(banknote)

        return players_win
