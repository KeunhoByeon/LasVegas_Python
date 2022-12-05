import time
import numpy as np
from .base_player import BasePlayer


class RuleBasePlayer(BasePlayer):
    def __init__(self, index, print_game: bool = True):
        super(RuleBasePlayer, self).__init__(index, print_game=print_game)

    def _make_input_tensor(self, game_info):
        """
        Input Size

        Game 1 * NumPlayer 1 Round 1 = 2
        Player 1 * Index 5 Money 1 Dice 6 WhiteDice 6 = 18
        Casinos 6 * Banknotes 5 Dice 5 = 60
        AllPlayers 5 * Money 1 DiceNum 1 WhiteDiceNum 1 = 15
        SUM = 95
        """
        players_info = game_info['players']
        casinos_info = game_info['casinos']
        round_index = game_info['round_index']
        num_players = len(players_info)

        input_array = []
        input_array.append(num_players / 5)
        input_array.append(round_index / 4)

        index_arr = [0 for _ in range(5)]
        index_arr[self.index - 1] = 1
        input_array.extend(index_arr)
        input_array.append(self._money / 10000 / 100)
        p_num_dice = [0 for _ in range(6)]
        p_num_white_dice = [0 for _ in range(6)]
        for d in self._dice:
            p_num_dice[d - 1] += 1
        for d in self._dice_white:
            p_num_white_dice[d - 1] += 1
        for i in range(6):
            p_num_dice[i] /= 8
            p_num_white_dice[i] /= 4
        input_array.extend(p_num_dice)
        input_array.extend(p_num_white_dice)

        for casino_index, casino_info in casinos_info.items():
            c_banknotes = []
            for banknote in casino_info['banknotes']:
                c_banknotes.append(banknote / 10000 / 10)
            for _ in range(5 - len(c_banknotes)):
                c_banknotes.append(0)

            c_dice = [0 for _ in range(5)]
            for die_index in casino_info['dice']:
                c_dice[die_index - 1] += 1
            for i in range(len(c_dice)):
                c_dice[i] /= 8

            input_array.extend(c_banknotes)
            input_array.extend(c_dice)

        for player_index in range(1, 6):
            if player_index in players_info.keys():
                player_info = players_info[player_index]
                input_array.append(player_info['money'] / 10000 / 100)
                input_array.append(player_info['num_dice'] / 8)
                input_array.append(player_info['num_dice_white'] / 4)
            else:
                input_array.extend([0, 0, 0])

        return input_array

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

    def _calc_expected_profit(self, dice_index, game_info, white_dice_value=0.5):
        num_players = len(game_info['players'])
        temp_dice_bucket = game_info['casinos'][dice_index]['dice']
        temp_banknotes_bucket = game_info['casinos'][dice_index]['banknotes']
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

        return money_changed, dice_value

    def _select_casino_rule_based(self, game_info, white_dice_value=0.5):
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
            money_changed, dice_value = self._calc_expected_profit(dice_index, game_info, white_dice_value)
            expected_profits[dice_index] = money_changed / dice_value

        for dice_index, money_changed in sorted(expected_profits.items(), key=lambda x: x[1], reverse=True):
            with open('./data/data_{}.csv'.format(time.strftime('%Y%m%d%H%M', time.localtime(time.time()))), 'a') as wf:
                wf.write(','.join(np.array(self._make_input_tensor(game_info)).astype(str)) + ',' + str(dice_index) + '\n')
            return dice_index

    def _select_casino(self, **kwargs):
        print("[Player{}]  Dice: {}{}".format(self.index, str(self._dice), str(self._dice_white))) if self._print_game else None
        select = self._select_casino_rule_based(game_info=kwargs["game_info"])
        print("Select Casino: {}".format(select)) if self._print_game else None
        return select
