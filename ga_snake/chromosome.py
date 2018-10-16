from pprint import pprint

from ga_snake import id_generator


class Chromosome(object):
    uid_gen = id_generator()

    def __init__(self, name: str, ancestors=None):
        self.uid = next(Chromosome.uid_gen)
        self.name = name
        self.fitness = None
        self.mutations = []
        self.ancestors = ancestors[::] if ancestors else []

    def shuffle(self):
        raise NotImplementedError()

    def clone(self, name=None):
        raise NotImplementedError()

    def create_child(self, name):
        assert name is not None
        child = self.clone(name)
        child.ancestors.append(self.name)
        return child

    def mutate(self, gene_mutation_prob):
        raise NotImplementedError()

    def show(self):
        pprint([
            'uid: {}'.format(self.uid),
            'name: {}'.format(self.name),
            'Fitness: {}'.format(self.fitness),
            'Ancestors: {}'.format(self.ancestors),
            'Mutations: {}'.format(self.mutations),
        ])

    @staticmethod
    def crossover(chromosome_1, chromosome_2, swap_prob: float):
        raise NotImplementedError()

    @staticmethod
    def new_random_chromosome(name: str, *args):
        raise NotImplementedError()
