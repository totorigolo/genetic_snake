import logging

from overrides import overrides

from ga_snake.population import Population
from ga_snake.training import Training

log = logging.getLogger("genetic_training")


class GeneticTraining(Training):
    def __init__(self, args):
        super().__init__(args)
        self.population = Population(args)
        self.max_generation = args.max_generation
        self.target_fitness = args.target_fitness

    @overrides
    def initial_batch(self):
        return self.population.chromosomes

    @overrides
    def create_batch_from_results(self, results):
        # Show the results
        results = sorted(results, key=lambda t: t[1])
        print('[Generation #%d] Results:' % self.population.generation)
        for uid, fitness, watch_link in results:
            print(' - #%3d fitness=%10g  => %s' % (uid, fitness, watch_link))

        # Checks if the target fitness is reached
        best_uid, max_fitness, _ = max(results, key=lambda r: r[1])
        if max_fitness >= self.target_fitness:
            log.info('Target fitness reached.')
            self.show_best()
            return []

        # Check that we don't exceed the maximum number of generations
        if self.population.generation >= self.max_generation:
            log.info('Maximum number of generations (%d) reached.',
                     self.max_generation)
            self.show_best()
            return []

        # Evolve the population and go on with the learning
        self.population.evolve({
            uid: fitness for uid, fitness, _ in results
        })
        return self.population.chromosomes

    def show_best(self):
        print('Best chromosome:\n', self.population.best_chromosome)
