from __future__ import annotations

from pprint import pprint
from typing import List

import numpy as np

from ga_snake import id_generator


def new_random_chromosome(name: str, layers: List[int]):
    chromosome = Chromosome(name, layers)
    chromosome.shuffle()
    return chromosome


class Chromosome(object):
    uid_gen = id_generator()

    def __init__(self, name: str, layers: List[int], ancestors=None):
        assert len(layers) >= 2
        self.uid = next(Chromosome.uid_gen)
        self.name = name
        self.layers = layers[::]
        self.layers_wb = []
        self.fitness = None
        self.mutations = []
        self.ancestors = ancestors[::] if ancestors else []

    @staticmethod
    def _random(shape=None):
        return np.random.normal(0, 30, shape)

    def shuffle(self):
        """ Fill the weights and biases with random values.
        """
        for i in range(len(self.layers) - 1):
            d0, d1 = self.layers[i], self.layers[i + 1]
            w = Chromosome._random([d0, d1])
            b = Chromosome._random([d1])
            self.layers_wb.append((w, b))

    def clone(self, name=None):
        if name is None:
            name = self.name
        clone = Chromosome(name, self.layers)
        clone.fitness = self.fitness
        clone.mutations = self.mutations[::]
        clone.ancestors = self.ancestors[::]
        for w, b in self.layers_wb:
            clone.layers_wb.append((np.copy(w), np.copy(b)))
        return clone

    def create_child(self, name):
        assert name is not None
        child = self.clone(name)
        child.ancestors.append(self.name)
        return child

    def mutate(self, gene_mutation_prob):
        for l, (w, b) in enumerate(self.layers_wb):
            for row in range(len(w)):
                for col in range(len(w[row])):
                    r = np.random.rand()
                    if r < gene_mutation_prob:
                        w[row][col] = Chromosome._random()
                        self.mutations.append('W{}-{}-{}'.format(l, row, col))
            for col in range(len(b)):
                row = np.random.rand()
                if row < gene_mutation_prob:
                    b[col] = Chromosome._random()
                    self.mutations.append('B{}-{}'.format(l, col))

    def show(self):
        pprint([
            'uid: {}'.format(self.uid),
            'name: {}'.format(self.name),
            'Layers: {}'.format(self.layers),
            'Fitness: {}'.format(self.fitness),
            'Ancestors: {}'.format(self.ancestors),
            'Mutations: {}'.format(self.mutations),
            {i: (w, b) for i, (w, b) in enumerate(self.layers_wb)},
            'Fitness: {}'.format(self.fitness),  # Repeated for convenience
        ])


def crossover(c1: Chromosome, c2: Chromosome, swap_prob: float):
    """ Uniform crossover.
    """
    for layer_idx in range(min(len(c1.layers_wb), len(c2.layers_wb))):
        # Swap w (?)
        r = np.random.rand()
        if r < swap_prob:
            w1, b1 = c1.layers_wb[layer_idx]
            w2, b2 = c2.layers_wb[layer_idx]
            c1.layers_wb[layer_idx] = w2, b1
            c2.layers_wb[layer_idx] = w1, b2
            c1.mutations.append('C{}-W{}'.format(c2.name, layer_idx))
            c2.mutations.append('C{}-W{}'.format(c1.name, layer_idx))

        # Swap b (?)
        r = np.random.rand()
        if r < swap_prob:
            w1, b1 = c1.layers_wb[layer_idx]
            w2, b2 = c2.layers_wb[layer_idx]
            c1.layers_wb[layer_idx] = w1, b2
            c2.layers_wb[layer_idx] = w2, b1
            c1.mutations.append('C{}-B{}'.format(c2.name, layer_idx))
            c2.mutations.append('C{}-B{}'.format(c1.name, layer_idx))
