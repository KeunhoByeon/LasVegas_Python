import copy
import multiprocessing as mp
import random

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from game_manager import GameManager
from player.ml_player import MLPlayer

GAME_NUM = 1000
RANDOM_SEED = 103
NUM_PROCESS = mp.cpu_count() - 1
PLAYER_SETTING = ["MLPlayer", "RuleBase", "Random", "RuleBase", "Random"]
# PLAYER_SETTING = ["MLPlayer", "Random", "Random", "Random", "Random"]
# PLAYER_SETTING = ["RuleBase", "RuleBase", "RuleBase", "RuleBase", "RuleBase"]
# PLAYER_SETTING = ["RuleBase", "Random", "Random", "Random", "Random"]

"""
PLAYER_SETTING = ["RuleBase", "Random", "Random"]
Player1 [7469, 1900, 631]
Player2 [1290, 4169, 4541]
Player3 [1241, 3931, 4828]
[3162, 13251, 13587]

PLAYER_SETTING = ["RuleBase", "RuleBase", "Random", "Random", "Random"]
Player1 [3805, 2391, 1796, 1203, 805]
Player2 [3234, 2711, 1912, 1273, 870]
Player3 [969, 1709, 2112, 2608, 2602]
Player4 [976, 1627, 2123, 2473, 2801]
Player5 [1016, 1562, 2057, 2443, 2922]
[12812, 13834, 24165, 24496, 24693]
"""

if __name__ == '__main__':
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    PLAYER_SETTINGS = []
    for s1 in ["RuleBase", "Random", None]:
        for s2 in ["RuleBase", "Random", None]:
            for s3 in ["RuleBase", "Random", None]:
                for s4 in ["RuleBase", "Random", None]:
                    for s5 in ["RuleBase", "Random", None]:
                        if "RuleBase" not in [s1, s2, s3, s4, s5]:
                            continue
                        none_count = 0
                        for s in [s1, s2, s3, s4, s5]:
                            if s is None:
                                none_count += 1
                        if none_count >= 4:
                            continue
                        PLAYER_SETTINGS.append([s1, s2, s3, s4, s5])
    for _ in range(100):
        PLAYER_SETTINGS.append(["RuleBase", "RuleBase", "RuleBase", "RuleBase", "RuleBase"])
    random.shuffle(PLAYER_SETTINGS)

    total_ranking = [[0 for _ in range(len(PLAYER_SETTING))] for _ in range(len(PLAYER_SETTING) + 1)]
    total_ranking_sum = [0 for _ in range(len(PLAYER_SETTING) + 1)]
    for PLAYER_SETTING in tqdm(PLAYER_SETTINGS):
        game_manager = GameManager(print_game=False)

        for slot_index, player_type in enumerate(PLAYER_SETTING):
            if player_type is None:
                continue
            game_manager.add_player(slot_index=slot_index + 1, player_type=player_type)

        if isinstance(game_manager.players_manager._player_slots[0], MLPlayer):
            game_manager.players_manager._player_slots[0].model.load_state_dict(torch.load('./results/20221129182255/0.pth'))

        rankings = Parallel(n_jobs=NUM_PROCESS)(delayed(copy.deepcopy(game_manager).run)() for _ in range(GAME_NUM))

        for ranking in rankings:
            for rank, player_index in enumerate(ranking):
                total_ranking[player_index][rank] += 1
                total_ranking_sum[player_index] += rank

    for i, line in enumerate(total_ranking):
        if i == 0:
            continue
        print('Player{} {}'.format(i, line))
    print(total_ranking_sum[1:])
