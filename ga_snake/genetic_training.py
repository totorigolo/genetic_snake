import datetime
import logging

from overrides import overrides

from ga_snake.population import Population
from ga_snake.training import Training

log = logging.getLogger("genetic_training")


class GeneticTraining(Training):
    def __init__(self, chromosome_class, args):
        super().__init__(args)
        self.population = Population(chromosome_class, args)
        self.max_generation = args.max_generation
        self.target_fitness = args.target_fitness

    @overrides
    def initial_batch(self):
        return self.population.chromosomes

    @overrides
    def create_batch_from_results(self, results):
        # Show the results
        sum_work = 0
        results = sorted(results, key=lambda t: t[2])
        print('[Generation #%d] Results:' % self.population.generation)
        for uid, name, fitness, work, watch_link in results:
            print(' - #%8s fitness=%10g W=%5d => %s' % (
                name, fitness, work, watch_link))
            sum_work += work

        # Checks if the target fitness is reached
        _, best_name, max_fitness, _, _ = max(results, key=lambda r: r[2])
        if max_fitness >= self.target_fitness:
            log.info('Target fitness reached by %s.', best_name)
            self.show_best()
            return [], sum_work

        # Check that we don't exceed the maximum number of generations
        if self.population.generation >= self.max_generation:
            log.info('Maximum number of generations (%d) reached.',
                     self.max_generation)
            self.show_best()
            return [], sum_work

        # Update the population with the results
        self.population.update({
            uid: fitness for uid, name, fitness, _, _ in results
        })

        # Dump the population to a file
        filename = 'dump-{}-gen-{}.txt'.format(
            datetime.datetime.now(), self.population.generation)
        with open(filename, 'w+') as file:
            self.population.dump_to(file)

        # Evolve the population
        self.population.evolve()

        # Return the next batch
        estimated_work = sum_work
        return self.population.chromosomes, estimated_work

    def training_interrupted(self):
        pass

    def show_best(self):
        if self.population.best_chromosome is not None:
            print('Best chromosome:')
            self.population.best_chromosome.show()
        else:
            print('No best chromosome.')
