import logging
import time

from ga_snake.snake_game_executor import SnakeGameExecutor

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
        global_start_time = time.time()
        with SnakeGameExecutor(self.args) as executor:
            batch = self.initial_batch()
            while len(batch) > 0:
                start_time = time.time()
                log.info('Running new batch: %d jobs.', len(batch))
                self.run_counter += len(batch)

                results = executor.run_batch(batch)

                batch_duration = time.time() - start_time
                log.info('Batch: %d simulations in %g sec.',
                         self.run_counter, batch_duration)

                batch = self.create_batch_from_results(results)

        training_duration = time.time() - global_start_time
        log.info('Trained with %d simulations in %g sec.',
                 self.run_counter, training_duration)
