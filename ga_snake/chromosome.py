from typing import List

import numpy as np


def new_random_chromosome(uid: int, layers: List[int]):
    chromosome = Chromosome(uid, layers)
    chromosome.shuffle()
    return chromosome


class Chromosome(object):
    def __init__(self, uid: int, layers: List[int]):
        assert len(layers) >= 2
        self.uid = uid
        self.layers = layers
        self.layers_wb = []
        self.fitness = None

    def shuffle(self):
        """ Fill the weights and biases with random values.
        """
        for i in range(len(self.layers) - 1):
            d0, d1 = self.layers[i], self.layers[i + 1]
            w = np.random.rand(d0, d1)
            b = np.random.rand(d1)
            self.layers_wb.append((w, b))

    def clone(self, uid):
        clone = Chromosome(uid, self.layers[::])
        clone.fitness = self.fitness
        for w, b in self.layers_wb:
            clone.layers_wb.append((np.copy(w), np.copy(b)))
        return clone

    def mutate(self, gene_mutation_prob):
        for w, b in self.layers_wb:
            for w_row in w:
                for i in range(len(w_row)):
                    r = np.random.rand()
                    if r < gene_mutation_prob:
                        w_row[i] = np.random.rand()


def crossover(c1: Chromosome, c2: Chromosome, swap_prob: float):
    """ Uniform crossover
    """
    for layer_idx in range(min(len(c1.layers_wb), len(c2.layers_wb))):
        # Swap w (?)
        r = np.random.rand()
        if r < swap_prob:
            w1, b1 = c1.layers_wb[layer_idx]
            w2, b2 = c2.layers_wb[layer_idx]
            c1.layers_wb[layer_idx] = w2, b1
            c2.layers_wb[layer_idx] = w1, b2

        # Swap b (?)
        r = np.random.rand()
        if r < swap_prob:
            w1, b1 = c1.layers_wb[layer_idx]
            w2, b2 = c2.layers_wb[layer_idx]
            c1.layers_wb[layer_idx] = w1, b2
            c2.layers_wb[layer_idx] = w2, b1
