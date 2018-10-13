import numpy as np

from ga_snake import id_generator
from ga_snake.chromosome import new_random_chromosome, crossover


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

        self.generation = 1
        self.chromosomes = [
            new_random_chromosome(next(self.id_gen), self.layers)
            for _ in range(self.population_size)
        ]

        self.best_chromosome = None

    def evolve(self, results):
        self.generation += 1
        self._selection(results)
        self._mutate_all()
        self._do_crossovers()

    def _selection(self, results):
        # Update the chromosomes fitness with the results
        chromosome_dict = {}
        for chromosome in self.chromosomes:
            chromosome_dict[chromosome.uid] = chromosome
            chromosome.fitness = results[chromosome.uid]

            # Save the best chromosome ever seen
            if (self.best_chromosome is None
                    or self.best_chromosome.fitness < chromosome.fitness):
                self.best_chromosome = chromosome.clone(chromosome.uid)

        # Compute each chromosome probability of being selected
        # based on its fitness. ([1] is fitness)
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
            next_population.append(chromosome_dict[ranking[0][1]])

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
        for chromosome in self.chromosomes:
            r = np.random.rand()
            if r < self.mutation_prob:
                chromosome.mutate(self.mutation_inner_prob)

    def _do_crossovers(self):
        children = []
        population_size = len(self.chromosomes)
        while population_size < self.population_size:
            # Randomly select two parent chromosomes
            father, mother = map(
                lambda idx: self.chromosomes[idx],
                np.random.random_integers(0, len(self.chromosomes) - 1, 2))

            # Clone the parents to create the children
            alice = father.clone(next(self.id_gen))
            bob = mother.clone(next(self.id_gen))

            # Cross-over (?)
            r = np.random.rand()
            if r < self.crossover_prob:
                crossover(alice, bob, self.crossover_uniform_prob)

            # Welcome!
            children.append(alice)
            children.append(bob)
            population_size += 2

        self.chromosomes.extend(children)

    @staticmethod
    def compute_probability(fitness, max_fitness):
        return fitness / max_fitness
