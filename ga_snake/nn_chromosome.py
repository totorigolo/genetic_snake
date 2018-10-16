import pprint
from typing import List

import numpy as np
from overrides import overrides

from ga_snake.chromosome import Chromosome


class NNChromosome(Chromosome):
    def __init__(self, name: str, layers: List[int], ancestors=None):
        super().__init__(name, ancestors)
        assert len(layers) >= 2
        self.layers = layers[::]
        self.layers_wb = []

    @staticmethod
    def _random(shape=None):
        return np.random.normal(0, 30, shape)

    @overrides
    def shuffle(self):
        """ Fill the weights and biases with random values.
        """
        for i in range(len(self.layers) - 1):
            d0, d1 = self.layers[i], self.layers[i + 1]
            w = NNChromosome._random([d0, d1])
            b = NNChromosome._random([d1])
            self.layers_wb.append((w, b))

    @overrides
    def clone(self, name=None):
        if name is None:
            name = self.name
        clone = NNChromosome(name, self.layers)
        clone.fitness = self.fitness
        clone.mutations = self.mutations[::]
        clone.ancestors = self.ancestors[::]
        for w, b in self.layers_wb:
            clone.layers_wb.append((np.copy(w), np.copy(b)))
        return clone

    @overrides
    def mutate(self, gene_mutation_prob):
        for l, (w, b) in enumerate(self.layers_wb):
            for row in range(len(w)):
                for col in range(len(w[row])):
                    r = np.random.rand()
                    if r < gene_mutation_prob:
                        w[row][col] = NNChromosome._random()
                        self.mutations.append('W{}-{}-{}'.format(l, row, col))
            for col in range(len(b)):
                row = np.random.rand()
                if row < gene_mutation_prob:
                    b[col] = NNChromosome._random()
                    self.mutations.append('B{}-{}'.format(l, col))

    @overrides
    def dump(self):
        return pprint.pformat({
            'uid': self.uid,
            'name': self.name,
            'Layers': self.layers,
            'Fitness': self.fitness,
            'Ancestors': self.ancestors,
            'Mutations': self.mutations,
            'WB': {i: (w, b) for i, (w, b) in enumerate(self.layers_wb)},
        })

    @staticmethod
    @overrides
    def crossover(c1, c2, swap_prob: float):
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

    @staticmethod
    @overrides
    def new_random_chromosome(name: str, *args):
        layers = args[0]
        chromosome = NNChromosome(name, layers)
        chromosome.shuffle()
        return chromosome
