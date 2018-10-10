import logging

from ga_snake.cygni import util

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


class Snake(object):
    def __init__(self):
        self.name = "DumbestEver"
        self.snake_id = None
        self.directions = [util.Direction.DOWN, util.Direction.UP, util.Direction.LEFT, util.Direction.RIGHT]

    def get_next_move(self, game_map):
        return util.Direction.RIGHT

    @staticmethod
    def on_game_ended():
        log.debug('The game has ended!')

    @staticmethod
    def on_snake_dead(reason):
        log.debug('Our snake died because %s', reason)

    @staticmethod
    def on_game_starting():
        log.debug('Game is starting!')

    def on_player_registered(self, snake_id):
        log.debug('Player registered successfully')
        self.snake_id = snake_id

    @staticmethod
    def on_invalid_player_name():
        log.fatal('Player name is invalid, try another!')

    @staticmethod
    def on_game_result(player_ranks):
        log.info('Game result:')
        for player in player_ranks:
            is_alive = 'alive' if player['alive'] else 'dead'
            log.info('%d. %d pts\t%s\t(%s)' %
                     (player['rank'], player['points'], player['playerName'],
                      is_alive))
