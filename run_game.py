import random

import numpy as np

from game_manager import GameManager

GAME_NUM = 100
RANDOM_SEED = 103
PLAYER_SETTING = ["Human", "RuleBase", "Random", "Random"]

if __name__ == '__main__':
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    game_manager = GameManager(print_game=True)

    for slot_index, player_type in enumerate(PLAYER_SETTING):
        game_manager.add_player(slot_index=slot_index + 1, player_type=player_type)

    for _ in range(GAME_NUM):
        ranking = game_manager.run()
        print("Ranking: {}".format(ranking))
