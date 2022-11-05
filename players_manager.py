from casinos_manager import CasinosManager
from player import EmptyPlayer, HumanPlayer, RuleBasePlayer


class PlayersManager:
    def __init__(self):
        self._player_slots = [EmptyPlayer(i + 1) for i in range(5)]

    def __len__(self):
        return len(self._player_slots)

    def __str__(self):
        # return "[PLAYERS]\n" + "\n".join((str(player) for player in self._player_slots))
        string = "[PLAYERS]"
        for player in self._player_slots:
            if isinstance(player, EmptyPlayer):
                continue
            string += '\n' + str(player)
        return string

    def reset_game(self):
        self.set_num_white_dice()
        for player in self._player_slots:
            player.reset_game()

    def reset_round(self):
        for player in self._player_slots:
            player.reset_round()

    def set_num_white_dice(self, num_white_dice: int = None):
        if num_white_dice is None:
            empty_cnt = 0
            for player in self._player_slots:
                if isinstance(player, EmptyPlayer):
                    empty_cnt += 1
            if empty_cnt == 0:
                num_white_dice = 0
            elif empty_cnt == 1 or empty_cnt == 2:
                num_white_dice = 2
            elif empty_cnt == 3:
                num_white_dice = 4
            else:
                num_white_dice = 8

        for player in self._player_slots:
            player.set_num_white_dice(num_white_dice)

    def add_player(self, slot_index: int, player_type: str = "Human"):
        if isinstance(self._player_slots[slot_index - 1], EmptyPlayer):
            if player_type == 'Human':
                self._player_slots[slot_index - 1] = HumanPlayer(slot_index)
                return True
            elif player_type == 'RuleBase':
                self._player_slots[slot_index - 1] = RuleBasePlayer(slot_index)
                return True
            else:
                print("Player type {} not yet implemented".format(player_type))
                return False
        else:
            print("Slot {} is not empty".format(slot_index))
            return False

    def del_player(self, slot_index: int):
        self._player_slots[slot_index - 1] = EmptyPlayer(slot_index)

    def add_banknotes(self, players_win: dict):
        for player_index, banknotes in players_win.items():
            if player_index == 0:
                continue
            self._player_slots[player_index - 1].add_banknotes(banknotes)

    def run_turn(self, casino_manager: CasinosManager, game_manager=None):
        all_done = True

        for player in self._player_slots:
            casino_index, dice = player.run_turn(game_manager=game_manager)
            if not casino_index:
                continue
            casino_manager.add_dice(casino_index=casino_index, dice=dice)
            all_done = False

        return all_done
