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
        sum_fitness = 0
        results = sorted(results, key=lambda t: t[2])
        print('[Generation #%d] Results:' % self.population.generation)
        for uid, name, fitness, watch_link in results:
            print(' - #%8s fitness=%10g  => %s' % (name, fitness, watch_link))
            sum_fitness += fitness

        # Checks if the target fitness is reached
        best_uid, best_name, max_fitness, _ = max(results, key=lambda r: r[2])
        if max_fitness >= self.target_fitness:
            log.info('Target fitness reached by %s.', best_name)
            self.show_best()
            return [], sum_fitness

        # Check that we don't exceed the maximum number of generations
        if self.population.generation >= self.max_generation:
            log.info('Maximum number of generations (%d) reached.',
                     self.max_generation)
            self.show_best()
            return [], sum_fitness

        # Evolve the population and go on with the learning
        self.population.evolve({
            uid: fitness for uid, name, fitness, _ in results
        })
        return self.population.chromosomes, sum_fitness

    def training_interrupted(self):
        pass

    def show_best(self):
        if self.population.best_chromosome is not None:
            print('Best chromosome:')
            self.population.best_chromosome.show()
        else:
            print('No best chromosome.')
