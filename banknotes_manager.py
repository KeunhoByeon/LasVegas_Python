import random

BANKNOTES_NUM = {10000: 6, 20000: 8, 30000: 8, 40000: 6, 50000: 6, 60000: 5, 70000: 5, 80000: 5, 90000: 5}


class BanknotesManager:
    def __init__(self):
        self._banknotes = []
        self.reset_game()

    def __len__(self):
        return len(self._banknotes)

    def __str__(self):
        return "[BANKNOTES]\n- Remain: {}".format(self.__len__())

    def reset_game(self):
        self._banknotes = []
        for price, num in BANKNOTES_NUM.items():
            for _ in range(num):
                self._banknotes.append(price)
        random.shuffle(self._banknotes)

    def reset_round(self):
        pass

    def get_banknote(self):
        return self._banknotes.pop(0)
