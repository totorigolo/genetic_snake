import logging

from ga_snake import util
from ga_snake.base_snake import BaseSnake
from ga_snake.util import Direction

log = logging.getLogger("snake")

TILE_VALUES = {
    util.TileType.EMPTY: 1,
    util.TileType.FOOD: 2,
    util.TileType.WALL: -1,
    util.TileType.OBSTACLE: -1,
    util.TileType.SNAKE_HEAD: -5,
    util.TileType.SNAKE_BODY: -1,
    util.TileType.SNAKE_TAIL: 5
}


class Snake(BaseSnake):
    def __init__(self, config):
        super().__init__()
        self.name = "DumbestEver"
        self.config = config
        self.result = None

        log.info("Running job: %s", self.config)

    @staticmethod
    def get_next_move(game_map):
        return Direction.RIGHT

    def on_game_result(self, player_ranks):
        def add_config(config):
            log.info("Adding config: %s", config)
            configs.append(config)

        # Generate some dummy configs
        configs = []
        if self.config < 20:
            add_config(self.config + 2)
            add_config(self.config + 4)
            add_config(self.config * 10)
            add_config((self.config + 1) * 10)

        self.result = configs
