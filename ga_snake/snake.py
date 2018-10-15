import logging
import os
from collections import deque
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
    util.TileType.SNAKE_TAIL: -5  # Because then it's protected 3 ticks.
}


def compute_distances(head: int, name: str, elements: List[int], gmap: Map):
    """ Compute the distances for each element of a given list.

    :param head: The (x,y) coordinates of the head of the snake.
    :param name: The name of the snake whose tail we don't want to eat.
    :param elements: A list of positions [pos].
    :param gmap: A game map.
    :return: The dict with the distances: {pos: dist}.
    """

    def coord_to_pos(_coord):
        if gmap.is_coordinate_out_of_bounds(_coord):
            return -1
        return util.translate_coordinate(_coord, gmap.width)

    def pos_to_coord(_pos):
        return util.translate_position(_pos, gmap.width)

    def is_free(_pos):
        return TILE_VALUES[
                   gmap.get_tile_at(pos_to_coord(_pos), name).tile_type] > 0

    inf = (gmap.width ** 2) / 2

    elements = {pos: inf for pos in elements}
    if len(elements) == 0:
        return elements
    if not is_free(head):  # If the head is in a wall/snake
        return elements
    num_seen_elements = 0

    # TODO: Use len(visited) to measure the open space

    visited = set()
    queue = deque([(head, 0)])
    while len(queue) > 0:
        pos, dist = queue.popleft()
        if pos in visited:
            continue
        visited.add(pos)

        # if dist > 6:
        #     break

        if pos in elements:
            elements[pos] = dist
            num_seen_elements += 1
            if num_seen_elements == len(elements):
                break

        x, y = pos_to_coord(pos)
        neighbors = filter(
            lambda p: p not in visited and is_free(p),
            map(coord_to_pos, [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
        )
        for neighbor in neighbors:
            queue.append((neighbor, dist + 1))

    # log.warning('food: %s', {
    #     pos_to_coord(pos): dist for pos, dist in elements.items()
    # })

    return elements


def abs_to_rel_quarters(current_direction, north, east, south, west,
                        debug=None):
    if current_direction == Direction.UP:
        front, right, back, left = north, east, south, west
    elif current_direction == Direction.RIGHT:
        front, right, back, left = east, south, west, north
    elif current_direction == Direction.LEFT:
        front, right, back, left = west, north, east, south
    else:  # DOWN
        front, right, back, left = south, west, north, east

    # if debug:
    #     log.warning('%s:'
    #                 '\n - direction: %s'
    #                 '\n - N=%g E=%g S=%g W=%g'
    #                 '\n - F=%g R=%g B=%g L=%g',
    #                 debug,
    #                 current_direction,
    #                 north, east, south, west,
    #                 front, right, back, left)

    return front, back, left, right


def get_quarter_distances_to_food(current_direction: Direction,
                                  head: int,
                                  name: str,
                                  food: List[int],
                                  gmap: Map):
    def coord_to_pos(_coord):
        if gmap.is_coordinate_out_of_bounds(_coord):
            return -1
        return util.translate_coordinate(_coord, gmap.width)

    inf = (gmap.width ** 2) / 2
    x, y = util.translate_position(head, gmap.width)

    head_north = coord_to_pos((x, y - 1))
    head_east = coord_to_pos((x + 1, y))
    head_south = coord_to_pos((x, y + 1))
    head_west = coord_to_pos((x - 1, y))

    food_north = compute_distances(head_north, name, food, gmap)
    food_east = compute_distances(head_east, name, food, gmap)
    food_south = compute_distances(head_south, name, food, gmap)
    food_west = compute_distances(head_west, name, food, gmap)

    north = min(food_north.items(), key=lambda fd: fd[1], default=(-1, inf))[1]
    east = min(food_east.items(), key=lambda fd: fd[1], default=(-1, inf))[1]
    south = min(food_south.items(), key=lambda fd: fd[1], default=(-1, inf))[1]
    west = min(food_west.items(), key=lambda fd: fd[1], default=(-1, inf))[1]

    # max_dist = max(north, east, south, west)
    north, east, south, west = map(lambda d: d / inf,
                                   [north, east, south, west])

    return abs_to_rel_quarters(current_direction,
                               north, east, south, west)

    # # Find the minimum distances for each quarter
    # north, east, south, west = inf, inf, inf, inf
    # for food, dist in foods_pos_dist.items():
    #     fx, fy = util.translate_position(food, gmap.width)
    #     fx, fy = fx - x, fy - y
    #     # if abs(fx) == abs(fy):
    #     #     continue  # Ignore food on the quarter edges
    #     sw = fx < fy  # The y coordinate is inverted
    #     se = -fx < fy
    #     if sw and se:
    #         south = min(south, dist)
    #     elif sw and not se:
    #         west = min(west, dist)
    #     elif not sw and se:
    #         east = min(east, dist)
    #     else:
    #         north = min(north, dist)
    #
    # # max_dist = max(north, east, south, west)
    # north, east, south, west = map(lambda d: d / inf,
    #                                [north, east, south, west])
    #
    # return abs_to_rel_quarters(current_direction,
    #                            north, east, south, west)


def get_quarter_num_food(current_direction: Direction,
                         head: int,
                         food_positions: List[int],
                         gmap: Map):
    x, y = util.translate_position(head, gmap.width)

    # Count the number of food per quarter
    north, east, south, west = 0, 0, 0, 0
    for food in food_positions:
        fx, fy = util.translate_position(food, gmap.width)
        fx, fy = fx - x, fy - y
        # if abs(fx) == abs(fy):
        #     continue  # Ignore food on the quarter edges
        sw = fx < fy  # The y coordinate is inverted
        se = -fx < fy
        if sw and se:
            south += 1
        elif sw and not se:
            west += 1
        elif not sw and se:
            east += 1
        else:
            north += 1

    return abs_to_rel_quarters(current_direction,
                               north, east, south, west)


def get_quarter_open_spaces(current_direction: Direction,
                            head: int,
                            name: str,
                            gmap: Map):
    def is_free(_coord):
        if gmap.is_coordinate_out_of_bounds(_coord):
            return False
        return TILE_VALUES[gmap.get_tile_at(_coord, name).tile_type] > 0

    x, y = util.translate_position(head, gmap.width)

    n = int(is_free((x, y - 1)))
    e = int(is_free((x + 1, y)))
    s = int(is_free((x, y + 1)))
    w = int(is_free((x - 1, y)))

    return abs_to_rel_quarters(current_direction, n, e, s, w, 'free')


class Snake(BaseSnake):
    def __init__(self, chromosome: Chromosome):
        super().__init__()
        self.name = "Darwin_%d" % chromosome.uid
        self.chromosome = chromosome
        self.result = None
        self.nn_created = False

        self.duration_bfs = 0

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

            if i < len(self.chromosome.layers) - 1:  # Input or hidden layers
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
        opponents = list(s['positions']
                         for s in gmap.game_map['snakeInfos'])
        # if s['name'] != self.name)
        # opponents_heads = list(o[0] for o in opponents)
        opponents_tails = list(o[-1] for o in opponents)
        # opponents = _flatten_list(opponents)

        # # Add the opponents' tails as food (yummy!)
        # food.extend(opponents_tails)

        # start_bfs = time.time()
        dist_food_quarters = get_quarter_distances_to_food(
            current_direction, head, self.name, food, gmap)
        # duration_bfs = time.time() - start_bfs
        # log.critical('BFS took: %g ms', duration_bfs * 1000)

        # num_food_quarters = get_quarter_num_food(
        #     current_direction, head, food, gmap)
        free_quarters = get_quarter_open_spaces(
            current_direction, head, self.name, gmap)

        return self._next_action_with_nn([
            # *num_food_quarters,
            *free_quarters,
            *dist_food_quarters,
        ])

    def _next_action_with_nn(self, inputs: List[int]):
        """ Use the NN to determine the next move given the inputs.
        """

        inputs = np.array(inputs)[np.newaxis]
        output = self.tf_session.run(
            self.nn_output, feed_dict={self.nn_inputs: inputs})

        # fblr
        # print(self.get_current_direction())
        # print(inputs)
        # k = 0
        # # ff = inputs[0][0+k]
        # # fb = inputs[0][1+k]
        # # fl = inputs[0][2+k]
        # # fr = inputs[0][3+k]
        # # k += 4
        # wf = inputs[0][0 + k]
        # wb = inputs[0][1 + k]
        # wl = inputs[0][2 + k]
        # wr = inputs[0][3 + k]
        # # k += 4
        # # df = inputs[0][0 + k]
        # # db = inputs[0][1 + k]
        # # dl = inputs[0][2 + k]
        # # dr = inputs[0][3 + k]
        #
        # # front = (1 - df) * .3 + wf * 1. + .0001
        # # left = (1 - dl) * .3 + wl * 1.
        # # right = (1 - dr) * .3 + wr * 1.
        #
        # front = wf * 1. + .0001
        # left = wl * 1.
        # right = wr * 1.
        #
        # # front = (1 - dl)
        # # left = (1 - df) + .0001
        # # right = (1 - dr)
        #
        # output[0][0] = front
        # output[0][1] = left
        # output[0][2] = right
        # print(output)

        return [Action.LEFT, Action.FRONT, Action.RIGHT][output.argmax()]

    def _compute_fitness(self, player_ranks):
        is_alive = None
        points = None
        # rank = None
        for player in player_ranks:
            if player['playerName'] == self.name:
                is_alive = player['alive']
                points = player['points']
                # rank = player['rank']

        alive_bonus = 3 if is_alive else 0
        num_food_eaten = (points - self.age / 3) / 3

        if is_alive:
            log.debug('Snake %s won :)', self.name)

        return self.age + points / 10_000 + alive_bonus
        # return points + self.age / 10_000 + alive_bonus
        # return num_food_eaten + points / 10_000 + alive_bonus

    @overrides
    def on_game_result(self, player_ranks):
        self.__close_tf()

        fitness = self._compute_fitness(player_ranks)
        self.result = self.chromosome.uid, fitness, self.watch_link
