import numpy as np

from .base_player import BasePlayer


class RuleBasePlayer(BasePlayer):
    def __init__(self, index, print_game: bool = True):
        super(RuleBasePlayer, self).__init__(index, print_game=print_game)

    def _calculate_temp_dice_order(self, temp_dice_bucket) -> list:
        temp_dice_unique, temp_dice_counts = np.unique(temp_dice_bucket, return_counts=True)

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

    def _calculate_temp_pay_out(self, temp_dice_bucket, temp_banknotes_bucket):
        players_win = {}

        dice_order = self._calculate_temp_dice_order(temp_dice_bucket)
        for banknote in sorted(temp_banknotes_bucket, reverse=True):
            if len(dice_order) == 0:
                break
            winner = dice_order.pop(0)
            if winner > 0:
                players_win[winner] = banknote

        return players_win

    def _select_casino_rule_based(self, game_manager, white_dice_value=0.5):
        available_options = []
        for d in self._dice:
            if d not in available_options:
                available_options.append(d)
        for d in self._dice_white:
            if d not in available_options:
                available_options.append(d)
        available_options.sort()

        expected_profits = {}
        for dice_index in available_options:
            num_players = game_manager.players_manager.get_num_players()
            temp_dice_bucket = game_manager.casinos_manager._casinos[dice_index - 1]._dice_bucket.copy()
            temp_banknotes_bucket = game_manager.casinos_manager._casinos[dice_index - 1]._banknotes_bucket.copy()
            original_payout = self._calculate_temp_pay_out(temp_dice_bucket, temp_banknotes_bucket)

            dice_value = 0
            for td in self._dice:
                if td == dice_index:
                    temp_dice_bucket.append(self.index)
                    dice_value += 1
            for td in self._dice_white:
                if td == dice_index:
                    temp_dice_bucket.append(0)
                    dice_value += white_dice_value
            new_payout = self._calculate_temp_pay_out(temp_dice_bucket, temp_banknotes_bucket)

            money_changed = 0
            player_indices = list(np.unique([list(original_payout.keys()) + list(new_payout.keys())]))
            for player_index in player_indices:
                player_money_changed = 0
                if player_index in original_payout.keys():
                    player_money_changed -= original_payout[player_index]
                if player_index in new_payout.keys():
                    player_money_changed += new_payout[player_index]
                if player_index == self.index:
                    money_changed += player_money_changed
                else:
                    money_changed -= player_money_changed / num_players
            expected_profits[dice_index] = money_changed / dice_value

        for dice_index, money_changed in sorted(expected_profits.items(), key=lambda x: x[1], reverse=True):
            return dice_index

    def _select_casino(self, **kwargs):
        print("[Player{}]  Dice: {}{}".format(self.index, str(self._dice), str(self._dice_white))) if self._print_game else None
        select = self._select_casino_rule_based(game_manager=kwargs["game_manager"])
        print("Select Casino: {}".format(select)) if self._print_game else None
        return select
