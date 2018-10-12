from ga_snake.training import Training


def _flatten_list(lst):
    if lst is None:
        return []
    return sum(lst, [])


class GeneticTraining(Training):
    def initial_batch(self):
        return [1]

    def create_batch_from_results(self, results):
        return _flatten_list(results)
