from game_manager import GameManager

GAME_NUM = 10

if __name__ == '__main__':
    game_manager = GameManager()
    game_manager.add_player(slot_index=1, player_type="Human")
    game_manager.add_player(slot_index=2, player_type="RuleBase")
    game_manager.add_player(slot_index=3, player_type="RuleBase")
    game_manager.add_player(slot_index=4, player_type="RuleBase")
    game_manager.add_player(slot_index=5, player_type="RuleBase")

    for _ in range(GAME_NUM):
        game_manager.run()
