import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_player import BasePlayer
from .ml.player_model import PlayerModel
from .rulebase_player import RuleBasePlayer


def get_target_tensor(prediction, target_is_real):
    target_tensor = torch.argmax(prediction.detach(), dim=2)
    target_tensor = F.one_hot(target_tensor, num_classes=6)
    if not target_is_real:
        target_tensor -= 1
        target_tensor *= -1
    target_tensor = target_tensor.type(torch.FloatTensor)
    return target_tensor


class MLPlayer(BasePlayer):
    def __init__(self, index: int, print_game: bool = True, train: bool = False, lr=0.0001):
        super(MLPlayer, self).__init__(index, print_game=print_game)
        self.training = train
        self.model = PlayerModel()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        if self.training:
            # TODO: TEMP Training Code
            self.loss_function = nn.CrossEntropyLoss()
            self.losses = []

            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=0.01)
            self.optimizer_available = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=0.01)
            self.memory_win_losses = []
            self.memory_loose_losses = []
            self.available_train = True
            self.cnt_rand = 0
            self.cnt_total = 0
            self.rule_base = RuleBasePlayer(index)
        else:
            self.model.eval()

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

        return torch.tensor([[input_array]], requires_grad=True)

    def _get_available_options(self):
        available_options = []
        for d in self._dice:
            if d not in available_options:
                available_options.append(d)
        for d in self._dice_white:
            if d not in available_options:
                available_options.append(d)
        available_options.sort()

        return available_options

    def _get_target_available(self, pred, available_options):
        target_available = torch.zeros_like(pred)
        for d_i in available_options:
            target_available[0][0][d_i - 1] = 1.
        target_available = target_available.type(torch.FloatTensor)
        return target_available

    def _get_target_rule_base(self, pred, game_info):
        self.rule_base._dice = self._dice
        self.rule_base._dice_white = self._dice_white
        target_index = self.rule_base._select_casino_rule_based(game_info)
        target_rule_base = torch.zeros_like(pred)
        target_rule_base[0][0][target_index - 1] = 1.
        target_rule_base = target_rule_base.type(torch.FloatTensor)
        return target_rule_base

    def _add_to_memory(self, pred, target_available, target_rule_base):
        target_better_option_win = torch.minimum(target_available, get_target_tensor(pred, True))
        if target_better_option_win.sum() > 0:
            # if torch.cuda.is_available():
            #     self.target_better_option_win = target_better_option_win.cuda()
            self.memory_win_losses.append(self.loss_function(pred, target_better_option_win[:, 0]))

        # if torch.cuda.is_available():
        #     self.target_rule_base = self.target_rule_base.cuda()
        self.memory_loose_losses.append(self.loss_function(pred, target_rule_base[:, 0]))

    def _select_casino_ml_model(self, game_info):
        available_options = self._get_available_options()
        input_tensor = self._make_input_tensor(game_info)

        if not self.training:
            with torch.no_grad():
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                pred = self.model(input_tensor)
                dice_order = torch.argsort(pred, dim=2, descending=True)[0][0]
                for dice_index in dice_order:
                    dice_index = int(dice_index.item() + 1)
                    if dice_index in available_options:
                        return dice_index

        self.cnt_total += 1
        while True:
            input_tensor = self._make_input_tensor(game_info)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            pred = self.model(input_tensor)
            dice_order = torch.argsort(pred, dim=2, descending=True)[0][0]
            dice_index = int(dice_order[0].item() + 1)
            # pred = torch.FloatTensor(torch.stack(pred))

            # # TODO: TEMP Training Code
            # self.rule_base._dice = self._dice
            # self.rule_base._dice_white = self._dice_white
            # target_index = self.rule_base._select_casino_rule_based(game_info)
            # if dice_index != target_index:
            #     target_rule_base = self._get_target_rule_base(pred, game_info)
            #     if torch.cuda.is_available():
            #         target_rule_base = target_rule_base.cuda()
            #     self.optimizer_available.zero_grad()
            #     loss = self.loss_function(pred[0], target_rule_base[0])
            #     loss.backward()
            #     self.optimizer_available.step()
            #     self.cnt_rand += 1
            #     self.losses.append(loss.item())
            #     continue
            # else:
            #     return dice_index
            # # TODO: Done

            target_available = self._get_target_available(pred, available_options)
            target_rule_base = self._get_target_rule_base(pred, game_info)

            if not self.available_train:
                self._add_to_memory(pred, target_available, target_rule_base)

            if dice_index in available_options:
                return dice_index

            self.cnt_rand += 1
            if self.available_train:
                if torch.cuda.is_available():
                    target_rule_base = target_rule_base.cuda()
                self.optimizer_available.zero_grad()
                train_target = (target_available + target_rule_base) / 2
                loss = self.loss_function(pred, train_target)
                loss.backward()
                self.optimizer_available.step()
                continue
            else:
                for dice_index in dice_order:
                    dice_index = int(dice_index.item() + 1)
                    if dice_index in available_options:
                        return dice_index
                    self.cnt_rand += 1

    def _select_casino(self, **kwargs):
        print("[Player{}]  Dice: {}{}".format(self.index, str(self._dice), str(self._dice_white))) if self._print_game else None
        select = self._select_casino_ml_model(game_info=kwargs["game_info"])
        print("Select Casino: {}".format(select)) if self._print_game else None
        return select
