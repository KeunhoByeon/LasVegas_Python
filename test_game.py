import copy
import multiprocessing as mp
import random

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from game_manager import GameManager
from player.ml_player import MLPlayer

GAME_NUM = 10000
RANDOM_SEED = 103
NUM_PROCESS = mp.cpu_count() - 1
# PLAYER_SETTING = ["MLPlayer", "Random"]
PLAYER_SETTING = ["MLPlayer", "Random", "Random", "Random", "Random"]

"""
PLAYER_SETTING = ["MLPlayer", "Random", "Random", "Random", "Random"]
Player1 [419, 218, 161, 134, 68]
Player2 [146, 197, 208, 220, 229]
Player3 [165, 209, 203, 200, 223]
Player4 [140, 193, 204, 202, 261]
Player5 [130, 183, 224, 244, 219]
[1214, 2189, 2107, 2251, 2239]

PLAYER_SETTING = ["MLPlayer", "Random"]
Player1 [739, 261]
Player2 [261, 739]
[261, 739]
"""

if __name__ == '__main__':
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    total_ranking = [[0 for _ in range(len(PLAYER_SETTING))] for _ in range(len(PLAYER_SETTING) + 1)]
    total_ranking_sum = [0 for _ in range(len(PLAYER_SETTING) + 1)]
    game_manager = GameManager(print_game=False)

    for slot_index, player_type in enumerate(PLAYER_SETTING):
        if player_type is None:
            continue
        game_manager.add_player(slot_index=slot_index + 1, player_type=player_type)

    if isinstance(game_manager.players_manager._player_slots[0], MLPlayer):
        game_manager.players_manager._player_slots[0].model.load_state_dict(torch.load('./results/20221206235003/model_best.pth', 'cpu').state_dict())

    rankings = Parallel(n_jobs=NUM_PROCESS)(delayed(copy.deepcopy(game_manager).run)() for _ in tqdm(range(GAME_NUM)))

    for ranking in rankings:
        for rank, player_index in enumerate(ranking):
            total_ranking[player_index][rank] += 1
            total_ranking_sum[player_index] += rank

    for i, line in enumerate(total_ranking):
        if i == 0:
            continue
        print('Player{} {}'.format(i, line))
    print(total_ranking_sum[1:])
