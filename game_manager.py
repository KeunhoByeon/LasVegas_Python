from banknotes_manager import BanknotesManager
from casinos_manager import CasinosManager
from players_manager import PlayersManager


class GameManager:
    def __init__(self, print_game: bool = True):
        self.players_manager = PlayersManager(print_game=print_game)
        self.casinos_manager = CasinosManager(print_game=print_game)
        self.banknotes_manager = BanknotesManager()
        self._print_game = print_game
        self.state = "READY"
        self.round = 0

    def __str__(self):
        return '\n\n\n\n' \
               '============================Las Vegas============================\n' \
               "{:>65}\n\n{}\n\n{}\n\n{}\n" \
               '================================================================='.format(self.state, self.banknotes_manager, self.casinos_manager, self.players_manager)

    def reset_game(self):
        self.state = "READY"
        self.casinos_manager.reset_game()
        self.banknotes_manager.reset_game()
        self.players_manager.reset_game()

    def reset_round(self):
        self.casinos_manager.reset_round()
        self.banknotes_manager.reset_round()
        self.players_manager.reset_round()

    def get_game_info(self):
        return {'round': self.round, 'casinos': self.casinos_manager.get_casinos_info(), 'players': self.players_manager.get_players_info(), 'text': str(self)}

    def add_player(self, slot_index, player_type: str = "Human"):
        self.players_manager.add_player(slot_index, player_type)

    def del_player(self, slot_index: int):
        self.players_manager.del_player(slot_index)

    def _run_round(self, round_index):
        self.round = round_index
        self.casinos_manager.set_banknotes(self.banknotes_manager)

        self.state = "ROUND {}".format(round_index)
        print(self) if self._print_game else None

        all_done = False
        while not all_done:
            all_done = self.players_manager.run_turn(self.casinos_manager, game_info=self.get_game_info())

        self.state = "ROUND {} DONE - CALCULATING".format(round_index)
        print(self) if self._print_game else None

        players_win = self.casinos_manager.pay_out()
        self.players_manager.add_banknotes(players_win)

        self.reset_round()
        self.state = "ROUND {} DONE".format(round_index)
        print(self) if self._print_game else None

    def run(self, rounds_num: int = 4):
        self.reset_game()

        self.state = "STARTING"
        print(self) if self._print_game else None

        for round_index in range(1, rounds_num + 1):
            self._run_round(round_index)
        self.round = 0

        self.state = "GAME DONE"
        print(self) if self._print_game else None

        return self.players_manager.get_ranking()
