from .base_player import BasePlayer
from .ml.player_model import PlayerModel


class MLPlayer(BasePlayer):
    def __init__(self, index, print_game: bool = True, train: bool = False):
        super(MLPlayer, self).__init__(index, print_game=print_game)
        self.training = train
        self.model = PlayerModel(train=train)

        if not self.training:
            self.model.eval()

    def _make_input_data(self, game_info):
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

        return [input_array]

    def _select_casino_ml_model(self, game_info):
        input_array = self._make_input_data(game_info)
        dice_index = self.model(input_array)
        del input_array
        return dice_index

    def _select_casino(self, **kwargs):
        print("[Player{}]  Dice: {}{}".format(self.index, str(self._dice), str(self._dice_white))) if self._print_game else None
        select = self._select_casino_ml_model(game_info=kwargs["game_info"])
        print("Select Casino: {}".format(select)) if self._print_game else None
        return select
