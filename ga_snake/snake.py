import logging
import os
from collections import deque
from typing import List

import numpy as np
import tensorflow as tf
from overrides import overrides

from ga_snake import util
from ga_snake.base_snake import BaseSnake, Action
from ga_snake.nn_chromosome import NNChromosome
from ga_snake.util import Direction, Map, TileType

log = logging.getLogger("snake")

TILE_VALUES = {
    TileType.EMPTY: 1,
    TileType.FOOD: 2,
    TileType.WALL: -1,
    TileType.OBSTACLE: -1,
    TileType.SNAKE_HEAD: -5,
    TileType.SNAKE_BODY: -1,
    TileType.SNAKE_TAIL: -5  # Because then it's then protected during 3 ticks.
}


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

    # return front, back, left, right
    return left, front, right


def info_from(head: int, name: str, gmap: Map, max_depth: int, nb_food: int):
    """ TODO

    :param head: The (x,y) coordinates of the head of the snake.
    :param name: The name of the snake whose tail we don't want to eat.
    :param gmap: A game map.
    :param max_depth: max distance for the BFS search.
    :return: TODO
    """

    def coord_to_pos(_coord):
        if gmap.is_coordinate_out_of_bounds(_coord):
            return -1
        return util.translate_coordinate(_coord, gmap.width)

    def pos_to_coord(_pos):
        return util.translate_position(_pos, gmap.width)

    def get_tile_type_at(_pos):
        return gmap.get_tile_at(pos_to_coord(_pos), name).tile_type

    def is_free(_pos):
        return TILE_VALUES[get_tile_type_at(_pos)] > 0

    max_dist = (gmap.width ** 2 + gmap.height ** 2) ** .5 + 1
    nb_cells = gmap.width * gmap.height

    size_accessible_area = 0
    num_accessible_food = 0
    sum_dist_enemy_heads = 0
    sum_dist_enemy_tails = 0
    min_dist_to_food = max_dist

    visited = set()
    queue = deque()
    if is_free(head):  # Don't search if the head is inside a wall/snake/...
        queue.append((head, 0))
    while len(queue) > 0:
        pos, dist = queue.popleft()
        if pos in visited:
            continue
        visited.add(pos)

        if dist > max_depth:
            break

        tile_type = get_tile_type_at(pos)
        if tile_type == TileType.EMPTY:
            size_accessible_area += 1
        # elif tile_type == TileType.WALL:
        #     pass
        elif tile_type == TileType.FOOD:
            num_accessible_food += 1
            min_dist_to_food = min(min_dist_to_food, dist)
        # elif tile_type == TileType.OBSTACLE:
        #     pass
        elif tile_type == TileType.SNAKE_HEAD:
            # get_tile_at() considers our head&tail as SNAKE_BODY
            sum_dist_enemy_heads += dist
        elif tile_type == TileType.SNAKE_TAIL:
            sum_dist_enemy_tails += dist
        # elif tile_type == TileType.SNAKE_BODY:
        #     pass

        x, y = pos_to_coord(pos)
        neighbors = filter(
            lambda p: p not in visited and is_free(p),
            map(coord_to_pos, [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
        )
        for neighbor in neighbors:
            queue.append((neighbor, dist + 1))

    return (
        size_accessible_area / nb_cells,
        num_accessible_food / nb_food,
        sum_dist_enemy_heads,
        sum_dist_enemy_tails,
        min_dist_to_food / max_dist
    )


def get_quarter_info(current_direction: Direction,
                     head: int,
                     name: str,
                     gmap: Map):
    def coord_to_pos(_coord):
        if gmap.is_coordinate_out_of_bounds(_coord):
            return -1
        return util.translate_coordinate(_coord, gmap.width)

    x, y = util.translate_position(head, gmap.width)

    nb_food = len(gmap.game_map['foodPositions'])
    nb_food = max(nb_food, 1)

    head_north = coord_to_pos((x, y - 1))
    head_east = coord_to_pos((x + 1, y))
    head_south = coord_to_pos((x, y + 1))
    head_west = coord_to_pos((x - 1, y))

    north = info_from(head_north, name, gmap, 20, nb_food)
    east = info_from(head_east, name, gmap, 20, nb_food)
    south = info_from(head_south, name, gmap, 20, nb_food)
    west = info_from(head_west, name, gmap, 20, nb_food)

    return abs_to_rel_quarters(current_direction, north, east, south, west)


class Snake(BaseSnake):
    def __init__(self, chromosome: NNChromosome):
        super().__init__()
        self.name = "Darwin_{}".format(chromosome.name)
        self.chromosome = chromosome
        self.result = None
        self.nn_created = False

        self.duration_bfs = 0
        self.fitness_heuristic = 0

        self.is_alive = None
        self.points = None

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
                nn_output = tf.nn.leaky_relu(tf.matmul(nn_output, tf_w) + tf_b)
            else:  # Output layer
                nn_output = tf.nn.softmax(tf.matmul(nn_output, tf_w) + tf_b)

        # self.nn_layers_wb[0] = (
        #     np.array([[-18.95296467, -2.13596425, 14.01769126],
        #               [19.66879924, 22.65781413, 6.27754393],
        #               [19.27465876, 20.167447, -23.73239392],
        #               [-22.31468644, -22.46992725, -13.65075968],
        #               [-7.39467312, 20.73225443, 6.34716963],
        #               [-19.41448364, -15.7094644, -12.37728874],
        #               [-1.40222571, -21.75790102, -5.07464144],
        #               [-2.69831108, -15.41660656, -2.1114672],
        #               [-9.97016183, -5.2808588, 7.62226732],
        #               [7.58305804, 18.1828687, 10.0396783],
        #               [14.32383273, 10.44323665, 8.82756336],
        #               [17.35717168, 3.50665331, 2.99022693],
        #               [-16.60894971, -16.96172354, -0.40144951],
        #               [22.97680445, 19.68220202, -18.96993012],
        #               [21.82363329, 15.62341413, -6.90265461]]),
        #     np.array([16.00013119, -10.99470738, 3.47454987])
        # )

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
            per_process_gpu_memory_fraction=0.4 / os.cpu_count())
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

    @overrides
    def get_next_action(self, gmap: Map):
        if not self.nn_created:
            # We can't initialize TensorFlow stuff in __init__ because
            # this object is Pickled to send it to a worker process
            self._init_nn()

        myself = gmap.get_snake_by_id(self.snake_id)['positions']
        head = myself[0]
        current_direction = self.get_current_direction()

        center = (gmap.width / 2, gmap.height / 2)
        head_coord = util.translate_position(head, gmap.width)
        dist_to_center = util.get_euclidian_distance(head_coord, center)
        max_dist = ((gmap.width ** 2 + gmap.height ** 2) ** .5) / 2
        v = 1 - (dist_to_center / max_dist) ** 1.4
        self.fitness_heuristic += max(v, 0)
        # print(v)

        quarters_info = get_quarter_info(
            current_direction, head, self.name, gmap)

        left = quarters_info[0]
        front = quarters_info[1]
        right = quarters_info[2]
        return self._next_action_with_nn([
            *left,
            *front,
            *right
        ])

    def _next_action_with_nn(self, inputs: List[int]):
        """ Use the NN to determine the next move given the inputs.
        """

        inputs = np.array(inputs)[np.newaxis]
        output = self.tf_session.run(
            self.nn_output, feed_dict={self.nn_inputs: inputs})

        # Forbid suicide
        left_free = inputs[0][5 * 0] > 0
        front_free = inputs[0][5 * 1] > 0
        right_free = inputs[0][5 * 2] > 0
        if not left_free:
            output[0][0] = float('-inf')
        if not front_free:
            output[0][1] = float('-inf')
        if not right_free:
            output[0][2] = float('-inf')

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

        alive_bonus = 500 if is_alive else 0
        # num_food_eaten = int(((points + 1) - self.age / 3) / 3)

        if is_alive:
            log.info('Snake %s won :)', self.name)

        # return self.age + points / 10000 + alive_bonus
        # return points + self.age / 10000 + alive_bonus
        # return num_food_eaten + self.age / 10000 + alive_bonus

        self.is_alive = is_alive
        self.points = points

        return ((self.fitness_heuristic / self.age) * points
                + self.age / 10000
                + alive_bonus)

    @overrides
    def on_game_result(self, player_ranks):
        self.__close_tf()

        fitness = self._compute_fitness(player_ranks)
        self.result = (self.chromosome.uid, self.chromosome.name,
                       fitness, {
                           'age': self.age,
                           'alive': self.is_alive,
                           'points': self.points
                       }, self.watch_link)
