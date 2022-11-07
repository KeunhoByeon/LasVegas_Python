from casinos_manager import CasinosManager
from player import EmptyPlayer, HumanPlayer, RuleBasePlayer, RandomPlayer


class PlayersManager:
    def __init__(self, print_game: bool = True):
        self._player_slots = [EmptyPlayer(index=i + 1) for i in range(5)]
        self._print_game = print_game

    def __len__(self):
        cnt = 0
        for _ in self._get_exist_players():
            cnt += 1
        return cnt

    def __str__(self):
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
                self._player_slots[slot_index - 1] = HumanPlayer(index=slot_index, print_game=self._print_game)
                return True
            elif player_type == 'RuleBase':
                self._player_slots[slot_index - 1] = RuleBasePlayer(index=slot_index, print_game=self._print_game)
                return True
            elif player_type == 'Random':
                self._player_slots[slot_index - 1] = RandomPlayer(index=slot_index, print_game=self._print_game)
                return True
            else:
                print("Player type {} not yet implemented".format(player_type))
                return False
        else:
            print("Slot {} is not empty".format(slot_index))
            return False

    def del_player(self, slot_index: int):
        self._player_slots[slot_index - 1] = EmptyPlayer(index=slot_index)

    def _get_exist_players(self):
        for player in self._player_slots:
            if not isinstance(player, EmptyPlayer):
                yield player

    def get_num_players(self):
        return len(self)

    def get_players_info(self):
        players_info = {}
        for player in self._get_exist_players():
            num_dice, num_dice_white = player.get_num_dice()
            players_info[player.index] = {'num_dice': num_dice, 'num_dice_white': num_dice_white, 'money': player.get_money()}
        return players_info

    def get_ranking(self):
        players_money = {}
        for player_i, player in enumerate(self._player_slots):
            if not isinstance(player, EmptyPlayer):
                players_money[player_i + 1] = player.get_money()

        ranking = []
        for player_index, player_money in sorted(players_money.items(), key=lambda x: x[1], reverse=True):
            ranking.append(player_index)
        return ranking

    def add_banknotes(self, players_win: dict):
        for player_index, banknotes in players_win.items():
            if player_index == 0:
                continue
            self._player_slots[player_index - 1].add_banknotes(banknotes)

    def run_turn(self, casinos_manager: CasinosManager, game_info=None):
        all_done = True

        for player in self._get_exist_players():
            casino_index, dice = player.run_turn(game_info=game_info)
            if not casino_index:
                continue
            casinos_manager.add_dice(casino_index=casino_index, dice=dice)
            game_info['casinos'] = casinos_manager.get_casinos_info()
            all_done = False

        return all_done
