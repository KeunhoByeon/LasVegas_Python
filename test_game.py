import copy
import multiprocessing as mp
import random

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from game_manager import GameManager

GAME_NUM = 10000
NUM_PROCESS = mp.cpu_count()
RANDOM_SEED = 103
PLAYER_SETTING = ["RuleBase", "RuleBase", "Random", "Random", "Random"]

"""
PLAYER_SETTING = ["RuleBase", "Random", "Random"]
Player1 [7526, 1799, 675]
Player2 [1263, 4252, 4485]
Player3 [1211, 3949, 4840]
[3149, 13222, 13629]

PLAYER_SETTING = ["RuleBase", "RuleBase", "Random", "Random", "Random"]
Player1 [3573, 2426, 1744, 1374, 883]
Player2 [3294, 2585, 1912, 1299, 910]
Player3 [1090, 1650, 2145, 2463, 2652]
Player4 [989, 1711, 2168, 2439, 2693]
Player5 [1054, 1628, 2031, 2425, 2862]
[13568, 13946, 23937, 24136, 24413]
"""

if __name__ == '__main__':
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    game_manager = GameManager(print_game=False)

    for slot_index, player_type in enumerate(PLAYER_SETTING):
        game_manager.add_player(slot_index=slot_index + 1, player_type=player_type)
    rankings = Parallel(n_jobs=NUM_PROCESS)(delayed(copy.deepcopy(game_manager).run)() for _ in tqdm(range(GAME_NUM)))

    total_ranking = [[0 for _ in range(len(PLAYER_SETTING))] for _ in range(len(PLAYER_SETTING) + 1)]
    total_ranking_sum = [0 for _ in range(len(PLAYER_SETTING) + 1)]
    for ranking in rankings:
        for rank, player_index in enumerate(ranking):
            total_ranking[player_index][rank] += 1
            total_ranking_sum[player_index] += rank

    for i, line in enumerate(total_ranking):
        if i == 0:
            continue
        print('Player{} {}'.format(i, line))
    print(total_ranking_sum[1:])
