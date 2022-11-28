# TODO: TEMP
import torch
import torch.nn as nn

import torch.nn.functional as F
from .base_player import BasePlayer
from .ml.player_model import PlayerModel


def get_target_tensor(prediction, target_is_real):
    target_tensor = torch.argmax(prediction.detach(), dim=2)
    target_tensor = F.one_hot(target_tensor, num_classes=6)
    if not target_is_real:
        target_tensor -= 1
        target_tensor *= -1
    target_tensor = target_tensor.type(torch.FloatTensor)
    return target_tensor


class MLPlayer(BasePlayer):
    def __init__(self, index, print_game: bool = True, train: bool = False):
        super(MLPlayer, self).__init__(index, print_game=print_game)
        self.training = train
        self.model = PlayerModel(train=train)
        self.available_train = True
        self.was_win = True
        self.cnt_rand = 0
        self.cnt_total = 0

        if not self.training:
            self.model.eval()

        # TODO: TEMP
        if self.training:
            self.loss_function = nn.BCEWithLogitsLoss()
            self.memory_preds = []
            self.memory_availables = []
            self.memory_win_losses = []
            self.memory_loose_losses = []

    def _make_input_tensor(self, game_info):
        """
        Input Size

        Game 1 * NumPlayer 1 Round 1 = 2
        Player 1 * Money 1 Dice 6 WhiteDice 6 = 13
        Casinos 6 * Banknotes 5 Dice 5 = 60
        OtherPlayers 4 * Money 1 DiceNum 1 WhiteDiceNum 1 = 12
        SUM = 87
        """
        players_info = game_info['players']
        casinos_info = game_info['casinos']
        round_index = game_info['round_index']
        num_players = len(players_info)

        input_array = []
        input_array.append(num_players / 5)
        input_array.append(round_index / 4)

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

            c_dice = [0 for _ in range(num_players)]
            for die_index in casino_info['dice']:
                c_dice[die_index - 1] += 1
            for i in range(len(c_dice)):
                c_dice[i] /= 60

            input_array.extend(c_banknotes)
            input_array.extend(c_dice)

        for player_index, player_info in players_info.items():
            if player_index == self.index:
                continue
            input_array.append(player_info['money'] / 10000 / 100)
            input_array.append(player_info['num_dice'] / 8)
            input_array.append(player_info['num_dice_white'] / 4)

        return torch.FloatTensor([input_array])

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

    def _add_to_memory(self, pred, target_available):
        target_better_option_win = torch.minimum(target_available, get_target_tensor(pred, True))
        if target_better_option_win.sum() > 0:
            self.memory_win_losses.append(self.loss_function(pred, target_better_option_win))

        target_better_option_loose = torch.minimum(target_available, get_target_tensor(pred, False))
        if target_better_option_loose.sum() > 0:
            self.memory_loose_losses.append(self.loss_function(pred, target_better_option_loose) / 5)

    def _select_casino_ml_model(self, game_info):
        available_options = self._get_available_options()
        input_tensor = self._make_input_tensor(game_info)

        if not self.training:
            with torch.no_grad():
                pred = self.model(input_tensor)
                dice_order = torch.argsort(pred, dim=1, descending=True)[0]
                for d in dice_order:
                    d = int(d.item() + 1)
                    if d in available_options:
                        return d

        self.cnt_total += 1
        while True:
            pred = self.model(self._make_input_tensor(game_info))
            dice_order = torch.argsort(pred, dim=1, descending=True)[0]
            dice_index = int(dice_order[0].item() + 1)
            pred = torch.FloatTensor(torch.stack([pred]))

            if dice_index in available_options:
                if not self.available_train:
                    target_available = self._get_target_available(pred, available_options)
                    self._add_to_memory(pred, target_available)
                return dice_index
            else:
                self.cnt_rand += 1
                if self.available_train:
                    target_available = self._get_target_available(pred, available_options)
                    self.model.optimizer.zero_grad()
                    loss = self.loss_function(pred, target_available) / len(available_options)
                    loss.backward()
                    self.model.optimizer.step()
                    continue
                else:
                    for dice_index in dice_order:
                        dice_index = int(dice_index.item() + 1)
                        if dice_index in available_options:
                            return dice_index

    def _select_casino(self, **kwargs):
        print("[Player{}]  Dice: {}{}".format(self.index, str(self._dice), str(self._dice_white))) if self._print_game else None
        select = self._select_casino_ml_model(game_info=kwargs["game_info"])
        print("Select Casino: {}".format(select)) if self._print_game else None
        return select
