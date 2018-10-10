import argparse
import asyncio
import json
import logging
import sys

import colorlog
from autobahn.asyncio.websocket import (WebSocketClientFactory,
                                        WebSocketClientProtocol)

from ga_snake.cygni import messages, util
from ga_snake.snake import Snake

log = logging.getLogger("client")
log_levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'fatal': logging.FATAL
}
log_names = list(log_levels)

loop = asyncio.get_event_loop()


class SnakebotProtocol(WebSocketClientProtocol):
    def __init__(self):
        super(WebSocketClientProtocol, self).__init__()

        self.snake = Snake()
        self.routing = {
            messages.GAME_ENDED: self._game_ended,
            messages.TOURNAMENT_ENDED: self._tournament_ended,
            messages.MAP_UPDATE: self._map_update,
            messages.SNAKE_DEAD: self._snake_dead,
            messages.GAME_STARTING: self._game_starting,
            messages.PLAYER_REGISTERED: self._player_registered,
            messages.INVALID_PLAYER_NAME: self._invalid_player_name,
            messages.HEART_BEAT_RESPONSE: self._heart_beat_response,
            messages.GAME_LINK_EVENT: self._game_link,
            messages.GAME_RESULT_EVENT: self._game_result
        }

    def onOpen(self):
        log.info("connection is open")
        self._send(messages.client_info())
        self._send(messages.player_registration(self.snake.name))

    def onMessage(self, payload, isBinary):
        assert not isBinary
        if isBinary:
            log.error('Received binary message, ignoring...')
            return

        msg = json.loads(payload.decode())
        log.debug("Message received: %s", msg)

        self._route_message(msg)

    def onClose(self, wasClean, code, reason):
        log.info("Socket is closed!")
        if reason:
            log.error(reason)

        if self.is_tournament:
            self.heart_beat.cancel()
        else:
            self._done(None)

    def _done(self, task):
        loop.stop()

    def _send(self, msg):
        log.debug("Sending message: %s", msg)
        self.sendMessage(json.dumps(msg).encode(), False)

    def _route_message(self, msg):
        fun = self.routing.get(msg['type'], None)
        if fun:
            fun(msg)
        else:
            self._unrecognied_message(msg)

    def _game_ended(self, msg):
        self.snake.on_game_ended()

        if not self.is_tournament:
            log.debug('Sending close message to websocket')
            self.sendClose()

        print("WATCH AT: ", self.watch_game_at)

    def _tournament_ended(self, msg):
        self.sendClose()

    def _map_update(self, msg):
        direction = self.snake.get_next_move(util.Map(msg['map']))
        self._send(messages.register_move(str(direction), msg))

    def _snake_dead(self, msg):
        self.snake.on_snake_dead(msg['deathReason'])

    def _game_starting(self, msg):
        self.snake.on_game_starting()

    def _player_registered(self, msg):
        self._send(messages.start_game())

        player_id = msg['receivingPlayerId']
        self.is_tournament = msg['gameMode'].upper() == 'TOURNAMENT'

        if self.is_tournament:
            self.heart_beat = loop.create_task(
                self._send_heart_beat(player_id))
            self.heart_beat.add_done_callback(self._done)

        self.snake.on_player_registered(player_id)

    def _invalid_player_name(self, msg):
        self.snake.on_invalid_player_name()

    def _heart_beat_response(self, msg):
        pass

    def _game_link(self, msg):
        log.info('Watch game at: %s', msg['url'])
        self.watch_game_at = msg["url"]

    def _game_result(self, msg):
        self.snake.on_game_result(msg['playerRanks'])

    def _unrecognied_message(self, msg):
        log.error('Received unrecognized message: %s', msg)

    async def _send_heart_beat(self, player_id):
        while True:
            self._send(messages.heart_beat(player_id))
            await asyncio.sleep(2)


def main():
    args = _parse_args()
    _set_up_logging(args)

    factory = WebSocketClientFactory(u"ws://%s:%s/%s" % (args.host, args.port,
                                                         args.venue))
    factory.protocol = SnakebotProtocol

    coro = loop.create_connection(factory, args.host, args.port)
    loop.run_until_complete(coro)

    loop.run_forever()
    loop.close()
    sys.exit(0)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Python client for Cygni's snakebot competition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-r',
        '--host',
        default='snake.cygni.se',
        help='The host to connect to')
    parser.add_argument(
        '-p', '--port', default='80', help='The port to connect to')
    parser.add_argument(
        '-v',
        '--venue',
        default='training',
        choices=['training', 'tournament'],
        help='The venue (training or tournament)')
    parser.add_argument(
        '-l',
        '--log-level',
        default=log_names[0],
        choices=log_names,
        help='The log level for the client')

    return parser.parse_args()


def _set_up_logging(args):
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        fmt=('%(log_color)s[%(asctime)s %(levelname)8s] --'
             ' %(message)s (%(filename)s:%(lineno)s)'),
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    log.addHandler(handler)
    log.setLevel(log_levels[args.log_level])
