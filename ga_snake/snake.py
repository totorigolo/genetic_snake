import logging
import os
from typing import List

import numpy as np
import tensorflow as tf
from overrides import overrides

from ga_snake import util
from ga_snake.base_snake import BaseSnake, Action
from ga_snake.chromosome import Chromosome
from ga_snake.util import Direction, Map

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


def abs_to_rel_quarters(current_direction, north, east, south, west):
    if current_direction == Direction.UP:
        front, right, back, left = north, east, south, west
    elif current_direction == Direction.RIGHT:
        front, right, back, left = east, south, west, north
    elif current_direction == Direction.LEFT:
        front, right, back, left = west, north, east, south
    else:  # DOWN
        front, right, back, left = south, west, north, east
    return back, front, left, right


def get_quarter_distances_to_food(current_direction: Direction,
                                  head: int,
                                  food_positions: List[int],
                                  gmap: Map):
    inf = gmap.width ** 3
    x, y = util.translate_position(head, gmap.width)

    # Find the minimum distances for each quarter
    north, east, south, west = inf, inf, inf, inf
    for food in food_positions:
        fx, fy = util.translate_position(food, gmap.width)
        dist = util.get_euclidian_distance((x, y), (fx, fy))
        fx, fy = fx - x, fy - y
        if abs(fx) == abs(fy):
            continue  # Ignore food on the quarter edges
        nw = fx < fy
        ne = -fx < fy
        if nw and ne:
            north = min(north, dist)
        elif nw and not ne:
            west = min(west, dist)
        elif not nw and ne:
            east = min(east, dist)
        else:
            south = min(south, dist)

    return abs_to_rel_quarters(current_direction,
                               north, east, south, west)


def get_quarter_open_spaces(current_direction: Direction,
                            head: int,
                            gmap: Map):
    x, y = util.translate_position(head, gmap.width)

    n = gmap.is_tile_available_for_movement((x, y + 1))
    e = gmap.is_tile_available_for_movement((x + 1, y))
    s = gmap.is_tile_available_for_movement((x, y - 1))
    w = gmap.is_tile_available_for_movement((x - 1, y))

    return abs_to_rel_quarters(current_direction, n, e, s, w)


class Snake(BaseSnake):
    def __init__(self, chromosome: Chromosome):
        super().__init__()
        self.name = "Darwin_%d" % chromosome.uid
        self.chromosome = chromosome
        self.result = None
        self.nn_created = False

    def _init_nn(self):
        """ Init the NN: create the TF flow from the chromosome.
        """
        num_inputs = self.chromosome.layers[0]
        nn_input = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs])

        nn_output = nn_input
        self.nn_layers_wb = []
        for i, (w, b) in enumerate(self.chromosome.layers_wb):
            tf_w = tf.Variable(w, dtype=tf.float32)
            tf_b = tf.Variable(b, dtype=tf.float32)
            self.nn_layers_wb.append((tf_w, tf_b))

            if i < len(self.chromosome.layers) - 1:  # Input ou hidden layers
                nn_output = tf.nn.relu(tf.matmul(nn_output, tf_w) + tf_b)
            else:  # Output layer
                nn_output = tf.nn.softmax(tf.matmul(nn_output, tf_w) + tf_b)

        self.nn_inputs = nn_input
        self.nn_output = nn_output

        self.__init_tf()

    def __init_tf(self):
        """ Create the TensorFlow session.
        """
        gpu_options = tf.GPUOptions(
            # By default, TensorFlow pre-allocates all the available
            # memory, so we can't run multiple processes at the same
            # time. Here, we ask not to pre-allocate to much.
            per_process_gpu_memory_fraction=0.5 / os.cpu_count())
        self.tf_session = tf.Session(config=tf.ConfigProto(
            use_per_session_threads=True,
            gpu_options=gpu_options
        ))
        self.tf_session.run(tf.global_variables_initializer())

        self.nn_created = True

    def __close_tf(self):
        """ Close the TensorFlow session.
        """
        self.tf_session.close()

    """
    NN inputs:
      * possibles?
        - current score
      * TBD:
        - food (advanced)
        - open spaces
        - opponents
      * done:
        - food (simple)
    """

    @overrides
    def get_next_action(self, gmap: Map):
        if not self.nn_created:
            # We can't initialize TensorFlow stuff in __init__ because
            # this object is Pickled to send it to a worker process
            self._init_nn()

        myself = gmap.get_snake_by_id(self.snake_id)['positions']
        head = myself[0]
        current_direction = self.get_current_direction()

        food = gmap.game_map['foodPositions']
        # obstacles = game_map.game_map['obstaclePositions']
        # opponents = list(s['positions']
        #                  for s in game_map.game_map['snakeInfos'])
        # opponents_heads = list(o[0] for o in opponents)
        # opponents_tails = list(o[-1] for o in opponents)
        # opponents = _flatten_list(opponents)

        food_quarters = get_quarter_distances_to_food(
            current_direction, head, food, gmap)
        free_quarters = get_quarter_open_spaces(current_direction, head, gmap)

        return self._next_action_with_nn([
            # *food_quarters,
            *free_quarters
        ])

    def _next_action_with_nn(self, inputs: List[int]):
        """ Use the NN to determine the next move given the inputs.
        """

        inputs = np.array(inputs)[np.newaxis]
        output = self.tf_session.run(
            self.nn_output, feed_dict={self.nn_inputs: inputs})

        return [Action.LEFT, Action.FRONT, Action.RIGHT][output.argmax()]

    def _compute_fitness(self, player_ranks):
        # is_alive = None
        points = None
        # rank = None
        for player in player_ranks:
            if player['playerName'] == self.name:
                # is_alive = player['alive']
                points = player['points']
                # rank = player['rank']
        return self.age + points / 100

    @overrides
    def on_game_result(self, player_ranks):
        self.__close_tf()

        fitness = self._compute_fitness(player_ranks)
        self.result = self.chromosome.uid, fitness, self.watch_link
