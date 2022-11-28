import gc
import multiprocessing as mp
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from game_manager import GameManager

EPOCH = 100
GAME_NUM = 100
RANDOM_SEED = 103
NUM_PROCESS = mp.cpu_count()
PLAYER_SETTING = ["MLPlayerTraining", "RuleBase", "RuleBase", "RuleBase", "Random"]
ML_PLAYER_INDEX = 1


def get_target_tensor(prediction, target_is_real):
    target_tensor = torch.argmax(torch.stack(prediction.copy()).detach(), dim=3)
    target_tensor = F.one_hot(target_tensor, num_classes=6)
    if not target_is_real:
        target_tensor -= 1
        target_tensor *= -1
    target_tensor = target_tensor.type(torch.FloatTensor)
    return target_tensor


if __name__ == '__main__':
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    game_manager = GameManager(print_game=False)
    for slot_index, player_type in enumerate(PLAYER_SETTING):
        game_manager.add_player(slot_index=slot_index + 1, player_type=player_type)

    for epoch in range(EPOCH):
        losses = []
        total_ranking_sum = [0 for _ in range(len(PLAYER_SETTING) + 1)]
        progress = tqdm(range(GAME_NUM), leave=False)
        for _ in progress:
            gc.collect()

            ml_player = game_manager.players_manager._player_slots[0]

            ml_player.cnt_rand, ml_player.cnt_total = 0, 0
            game_manager.players_manager._player_slots[0].available_train = True
            ranking = game_manager.run()
            cnt_1 = round(ml_player.cnt_rand / ml_player.cnt_total, 4)

            ml_player.cnt_rand, ml_player.cnt_total = 0, 0
            game_manager.players_manager._player_slots[0].available_train = False
            ranking = game_manager.run()
            cnt_2 = round(ml_player.cnt_rand / ml_player.cnt_total, 4)

            result = True if ranking[0] == ML_PLAYER_INDEX else False

            # winner_money = game_manager.players_manager.get_players_info()[ranking[0]]['money']
            # my_money = game_manager.players_manager.get_players_info()[ML_PLAYER_INDEX]['money']

            # targets_available = torch.stack(ml_player.memory_availables)
            # targets_result = get_target_tensor(ml_player.memory_preds, result)
            # targets = torch.minimum(targets_available, targets_result)

            ml_player.model.optimizer.zero_grad()
            loss = torch.sum(torch.stack(ml_player.memory_win_losses if result else ml_player.memory_loose_losses))
            loss.backward()
            ml_player.model.optimizer.step()
            ml_player.memory_win_losses = []
            ml_player.memory_loose_losses = []

            for rank, player_index in enumerate(ranking):
                total_ranking_sum[player_index] += rank

            progress.set_description(desc='Loss: {}\t{}  Cnt1: {}  Cnt2: {}'.format(round(loss.item(), 8), str(total_ranking_sum[1:]), cnt_1, cnt_2))
        print('Epoch {}\tLoss: {}\t{}'.format(epoch, loss, str(total_ranking_sum[1:])))

# Old Training Sequence
# winner_money = game_manager.players_manager.get_players_info()[ranking[0]]['money']
# my_money = game_manager.players_manager.get_players_info()[ml_training_player_index + 1]['money']
# result = my_money / winner_money if winner_money > 0 else 0
# game_manager.players_manager._player_slots[0].model.optimize_parameters(result=result)
# loss = game_manager.players_manager._player_slots[0].model.loss.detach().numpy()
# loss = game_manager.players_manager._player_slots[0].model.loss
# losses.append(np.mean(loss))
# game_manager.players_manager._player_slots[0].model.loss = []
