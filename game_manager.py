from banknotes_manager import BanknotesManager
from casinos_manager import CasinosManager
from players_manager import PlayersManager


class GameManager:
    def __init__(self):
        self.players_manager = PlayersManager()
        self.casino_manager = CasinosManager()
        self.banknotes_manager = BanknotesManager()
        self.state = "READY"

    def __str__(self):
        return '\n\n\n\n' \
               '============================Las Vegas============================\n' + \
               "{:>65}\n\n{}\n\n{}\n\n{}\n".format(self.state, self.banknotes_manager, self.casino_manager, self.players_manager) + \
               '================================================================='

    def reset_game(self):
        self.state = "READY"
        self.casino_manager.reset_game()
        self.banknotes_manager.reset_game()
        self.players_manager.reset_game()

    def reset_round(self):
        self.casino_manager.reset_round()
        self.banknotes_manager.reset_round()
        self.players_manager.reset_round()

    def add_player(self, slot_index, player_type: str = "Human"):
        self.players_manager.add_player(slot_index, player_type)

    def del_player(self, slot_index: int):
        self.players_manager.del_player(slot_index)

    def _run_round(self, round_index):
        self.casino_manager.set_banknotes(self.banknotes_manager)

        self.state = "ROUND {}".format(round_index)
        print(self)

        all_done = False
        while not all_done:
            all_done = self.players_manager.run_turn(self.casino_manager, game_manager=self)

        self.state = "ROUND {} DONE - CALCULATING".format(round_index)
        print(self)

        players_win = self.casino_manager.pay_out()
        self.players_manager.add_banknotes(players_win)

        self.reset_round()
        self.state = "ROUND {} DONE".format(round_index)
        print(self)

    def run(self, rounds_num: int = 4):
        self.reset_game()

        self.state = "STARTING"
        print(self)

        for round_index in range(1, rounds_num + 1):
            self._run_round(round_index)

        self.state = "GAME DONE"
        print(self)

