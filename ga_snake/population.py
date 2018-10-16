import logging

import numpy as np

from ga_snake import id_generator
from ga_snake.chromosome import new_random_chromosome, crossover

log = logging.getLogger("population")


class Population(object):
    def __init__(self, args):
        self.id_gen = id_generator()

        self.layers = args.layers
        self.population_size = args.population_size

        self.elitism = args.selection_elitism
        self.selection_rank_prob = args.selection_rank_prob
        self.selection_keep_frac = args.selection_keep_frac

        self.mutation_prob = args.mutation_prob
        self.mutation_inner_prob = args.mutation_inner_prob

        self.crossover_prob = args.crossover_prob
        self.crossover_uniform_prob = args.crossover_uniform_prob
        self.num_new_random_per_generation = args.num_new_random_per_generation

        self.generation = 1
        self.chromosomes = [
            new_random_chromosome(self._new_name(), self.layers)
            for _ in range(self.population_size)
        ]

        self.best_chromosome = None

    def _new_name(self):
        return '{}-{}'.format(self.generation, next(self.id_gen))

    def evolve(self, results):
        log.info('Evolving generation %d => %d...',
                 self.generation, self.generation + 1)

        self.generation += 1
        self._selection(results)
        self._mutate_all()
        self._do_crossovers()

        # Add some totally random chromosomes
        for _ in range(self.num_new_random_per_generation):
            self.chromosomes.append(
                new_random_chromosome(self._new_name(), self.layers))
            self.chromosomes[-1].ancestors.append('R')  # Late random

        # Log the best chromosome seen so far
        if self.best_chromosome is not None:
            print('Best so far:')
            self.best_chromosome.show()

            # If elitist, add the best seen to the population
            if self.elitism:
                self.chromosomes.append(self.best_chromosome.clone())

        log.info('Generation %d is ready!', self.generation)

    def _selection(self, results):
        log.debug('Performing selection...')

        # Update the chromosomes fitness with the results
        chromosome_dict = {}
        for chromosome in self.chromosomes:
            chromosome_dict[chromosome.uid] = chromosome
            chromosome.fitness = results[chromosome.uid]

            # Save the best chromosome ever seen
            if (self.best_chromosome is None
                    or self.best_chromosome.fitness < chromosome.fitness):
                self.best_chromosome = chromosome.clone()

        # Compute each chromosome probability of being selected
        # based on its fitness. (kv[1] is fitness)
        ranking = list(enumerate(map(
            lambda kv: kv[0],  # Discard the fitness
            sorted(results.items(), key=lambda kv: kv[1], reverse=True)
        )))
        p_sel = self.selection_rank_prob
        probabilities = list(
            (uid, p_sel * (1 - p_sel) ** rank)
            for rank, uid in ranking
        )

        # Adjust the worst chromosome's probability to sum to 1.
        # (depending on p_sel, the worst can be ranked better than last)
        probabilities[-1] = (  # tuple don't support assignement
            probabilities[-1][0], (1 - p_sel) ** len(probabilities))

        # TODO: Compute rank with fitness AND variance

        # Elitism: Keep the best chromosome
        next_population = []
        if self.elitism:
            best = chromosome_dict[ranking[0][1]]
            next_population.append(best)

        # Roulette-wheel selection
        number_to_keep = int(self.selection_keep_frac * self.population_size
                             - 1 if self.elitism else 0)
        for _ in range(number_to_keep):
            r = np.random.rand()
            selected = None
            for uid, prob in probabilities:
                if r < prob:
                    selected = chromosome_dict[uid]
                    break
                r -= prob
            if selected is None:  # Rounding error
                selected = chromosome_dict[probabilities[-1][0]]

            next_population.append(selected)
        self.chromosomes = next_population

    def _mutate_all(self):
        log.debug('Performing mutations...')

        for chromosome in self.chromosomes:
            r = np.random.rand()
            if r < self.mutation_prob:
                chromosome.mutate(self.mutation_inner_prob)

    def _do_crossovers(self):
        log.debug('Performing crossovers...')

        children = []
        population_size = len(self.chromosomes)
        while population_size < self.population_size:
            # Randomly select two parent chromosomes
            father, mother = map(
                lambda idx: self.chromosomes[idx],
                np.random.random_integers(0, len(self.chromosomes) - 1, 2))

            # Clone the parents to create the children
            alice = father.create_child(self._new_name())
            bob = mother.create_child(self._new_name())

            # Cross-over (?)
            r = np.random.rand()
            if r < self.crossover_prob:
                alice.ancestors.append('C{}'.format(mother.uid))
                bob.ancestors.append('C{}'.format(father.uid))
                crossover(alice, bob, self.crossover_uniform_prob)

            # Welcome!
            children.append(alice)
            children.append(bob)
            population_size += 2

        self.chromosomes.extend(children)

    @staticmethod
    def compute_probability(fitness, max_fitness):
        return fitness / max_fitness
