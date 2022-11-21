import multiprocessing as mp
import random
import gc
import numpy as np
from tqdm import tqdm

from game_manager import GameManager

EPOCH = 100
GAME_NUM = 100
RANDOM_SEED = 103
NUM_PROCESS = mp.cpu_count()
PLAYER_SETTING = ["MLPlayerTraining", "RuleBase", "RuleBase", "RuleBase", "Random"]

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
            ranking = game_manager.run()
            winner_money = game_manager.players_manager.get_players_info()[ranking[0]]['money']
            my_money = game_manager.players_manager.get_players_info()[1]['money']
            result = my_money / winner_money if winner_money > 0 else 0
            game_manager.players_manager._player_slots[0].model.optimize_parameters(result=result)
            loss = game_manager.players_manager._player_slots[0].model.loss.detach().numpy()
            losses.append(loss)
            for rank, player_index in enumerate(ranking):
                total_ranking_sum[player_index] += rank
            progress.set_description(desc='Loss: {}\t'.format(loss) + str(total_ranking_sum[1:]))
        print('Epoch {}\tLoss: {}\t{}'.format(epoch, np.mean(losses), str(total_ranking_sum[1:])))
