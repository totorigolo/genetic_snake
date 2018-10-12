import logging

from ga_snake.executor import Executor

log = logging.getLogger("training")


class Training(object):
    def __init__(self, args):
        self.args = args
        self.run_counter = 0

    def initial_batch(self):
        raise NotImplementedError()

    def create_batch_from_results(self, results):
        raise NotImplementedError()

    def train(self):
        with Executor(self.args) as executor:
            batch = self.initial_batch()
            while len(batch) > 0:
                log.info('Running batch: %s', batch)
                self.run_counter += len(batch)

                results = executor.run_batch(batch)
                batch = self.create_batch_from_results(results)

        log.info('Trained with %d runs.', self.run_counter)
