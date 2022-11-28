import gc
import multiprocessing as mp
import os
import random
import time

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
    result_dir = os.path.join('./results', time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    game_manager = GameManager(print_game=False)
    for slot_index, player_type in enumerate(PLAYER_SETTING):
        game_manager.add_player(slot_index=slot_index + 1, player_type=player_type)

    ml_player = game_manager.players_manager._player_slots[0]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ml_player.model.optimizer, T_max=10, eta_min=0.00001)

    for epoch in range(EPOCH):
        losses, cnts_1, cnts_2 = [], [], []
        rank_sum_1 = [0 for _ in range(len(PLAYER_SETTING) + 1)]
        rank_sum_2 = [0 for _ in range(len(PLAYER_SETTING) + 1)]
        progress = tqdm(range(GAME_NUM), leave=True)
        for _ in progress:
            gc.collect()

            # Main Training
            ml_player.cnt_rand, ml_player.cnt_total = 0, 0
            game_manager.players_manager._player_slots[0].available_train = False
            ranking = game_manager.run()
            cnt_1 = round(ml_player.cnt_rand / ml_player.cnt_total, 4)

            for rank, player_index in enumerate(ranking):
                rank_sum_1[player_index] += rank

            result = True if ranking[0] == ML_PLAYER_INDEX else False

            winner_money = game_manager.players_manager.get_players_info()[ranking[0]]['money']
            my_money = game_manager.players_manager.get_players_info()[ML_PLAYER_INDEX]['money']
            total_money = 0
            for p_index, p_info in game_manager.players_manager.get_players_info().items():
                total_money += p_info['money']

            ml_player.model.optimizer.zero_grad()
            loss = torch.mean(torch.stack(ml_player.memory_win_losses if result else ml_player.memory_loose_losses))
            loss = loss if result else loss * (1 + my_money / winner_money)
            loss = loss if not result else loss * (1 + my_money / total_money)
            loss.backward()
            ml_player.model.optimizer.step()
            ml_player.memory_win_losses = []
            ml_player.memory_loose_losses = []

            # Available Training
            ml_player.cnt_rand, ml_player.cnt_total = 0, 0
            game_manager.players_manager._player_slots[0].available_train = True
            ranking = game_manager.run()
            cnt_2 = round(ml_player.cnt_rand / ml_player.cnt_total, 4)

            for rank, player_index in enumerate(ranking):
                rank_sum_2[player_index] += rank

            losses.append(loss.item())
            cnts_1.append(cnt_1)
            cnts_2.append(cnt_2)

            current_loss = round(np.mean(losses), 4)
            current_rank_1 = str(np.round(np.array(rank_sum_1[1:]) / np.sum(rank_sum_1[1:]) * 100))
            current_rank_2 = str(np.round(np.array(rank_sum_2[1:]) / np.sum(rank_sum_2[1:]) * 100))
            current_cnt_1 = round(np.mean(cnts_1), 2)
            current_cnt_2 = round(np.mean(cnts_2), 2)
            line = '[Epoch {}]  Loss: {}  Ranking1: {}  Ranking2: {}  Cnt1: {}  Cnt2: {}'.format(
                epoch, current_loss, current_rank_1, current_rank_2, current_cnt_1, current_cnt_2)
            progress.set_description(line)

        scheduler.step()

        if epoch % 10 == 0:
            os.makedirs(result_dir, exist_ok=True)
            torch.save(ml_player.model.state_dict(), os.path.join(result_dir, '{}.pth'.format(epoch)))
