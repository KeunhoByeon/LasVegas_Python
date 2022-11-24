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
        Player 1 * Money 1 Dice 8 WhiteDice 4 = 13
        Casinos 6 * Banknotes 5 Dice 5 = 60
        OtherPlayers 4 * Money 1 DiceNum 1 WhiteDiceNum 1 = 12
        SUM = 87
        """
        players_info = game_info['players']
        casinos_info = game_info['casinos']
        round_index = game_info['round_index']
        num_players = len(players_info)

        input_array = []
        input_array.append(num_players)
        input_array.append(round_index)

        input_array.append(self._money)
        p_dice = [0 for _ in range(8)]
        for i, d in enumerate(self._dice):
            p_dice[i] = d
        p_dice_white = [0 for _ in range(4)]
        for i, d in enumerate(self._dice_white):
            p_dice_white[i] = d
        input_array.extend(p_dice)
        input_array.extend(p_dice_white)

        for casino_index, casino_info in casinos_info.items():
            c_banknotes = []
            for banknote in casino_info['banknotes']:
                c_banknotes.append(banknote / 10000 / 10)
            for _ in range(5 - len(c_banknotes)):
                c_banknotes.append(0)

            c_dice = [0 for _ in range(num_players)]
            for die_index in casino_info['dice']:
                c_dice[die_index - 1] += 1

            input_array.extend(c_banknotes)
            input_array.extend(c_dice)

        for player_index, player_info in players_info.items():
            if player_index == self.index:
                continue
            input_array.append(player_info['money'])
            input_array.append(player_info['num_dice'])
            input_array.append(player_info['num_dice_white'])

        return torch.FloatTensor([input_array])

    def _select_casino_ml_model(self, game_info):
        available_options = []
        for d in self._dice:
            if d not in available_options:
                available_options.append(d)
        for d in self._dice_white:
            if d not in available_options:
                available_options.append(d)
        available_options.sort()
        input_tensor = self._make_input_tensor(game_info)

        if not self.training:
            with torch.no_grad():
                pred = self.model(input_tensor)
                dice_index = int(torch.argmax(pred, dim=1)[0].item() + 1)
                return dice_index if dice_index in available_options else available_options[0]

        while True:
            pred = self.model(self._make_input_tensor(game_info))
            dice_index = int(torch.argmax(pred, dim=1)[0].item() + 1)

            pred = torch.FloatTensor(torch.stack([pred]))
            target = torch.zeros_like(pred)
            for d_i in available_options:
                target[0][0][d_i - 1] = 1.
            target = target.type(torch.FloatTensor)

            if dice_index not in available_options:
                self.model.optimizer.zero_grad()
                loss = self.loss_function(pred, target) / len(available_options)
                loss.backward()
                self.model.optimizer.step()
                continue

            if not self.available_train:
                self.memory_preds.append(pred)
                self.memory_availables.append(target)

                self.model.optimizer.zero_grad()
                self.loss = self.loss_function(pred, torch.minimum(target, get_target_tensor(pred, self.was_win)))
                self.loss.backward()
                self.model.optimizer.step()

            return dice_index

    def _select_casino(self, **kwargs):
        print("[Player{}]  Dice: {}{}".format(self.index, str(self._dice), str(self._dice_white))) if self._print_game else None
        select = self._select_casino_ml_model(game_info=kwargs["game_info"])
        print("Select Casino: {}".format(select)) if self._print_game else None
        return select
