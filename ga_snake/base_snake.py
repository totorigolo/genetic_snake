import logging

log = logging.getLogger("base_snake")


class BaseSnake(object):
    def __init__(self):
        self.name = None
        self.snake_id = None

    def on_player_registered(self, snake_id):
        self.snake_id = snake_id
        log.debug('Player "%s:%d" registered successfully',
                  self.name, self.snake_id)

    @staticmethod
    def on_game_starting():
        log.debug('Game is starting!')

    @staticmethod
    def on_snake_dead(reason):
        log.debug('Our snake died because %s', reason)

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

    @staticmethod
    def on_game_ended():
        log.debug('The game has ended!')
