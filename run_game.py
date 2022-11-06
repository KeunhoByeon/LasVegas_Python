import random

import numpy as np

from game_manager import GameManager

GAME_NUM = 10
RANDOM_SEED = 103

if __name__ == '__main__':
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    game_manager = GameManager()
    game_manager.add_player(slot_index=1, player_type="Human")
    game_manager.add_player(slot_index=2, player_type="RuleBase")
    game_manager.add_player(slot_index=3, player_type="Random")
    game_manager.add_player(slot_index=4, player_type="Random")

    for _ in range(GAME_NUM):
        game_manager.run()
