import logging
import os
import signal
import time

from ga_snake.snake_game_executor import SnakeGameExecutor

log = logging.getLogger("training")


class Training(object):
    def __init__(self, args):
        self.args = args
        self.run_counter = 0

        self.main_pid = None
        self.execution_stopped = False

    def initial_batch(self):
        raise NotImplementedError()

    def create_batch_from_results(self, results):
        """ Create the next batch and return an estimate of the work
        done in the previous batch.

        :param results: the list of all the results for the previous batch
        :return: (new_batch, estimate_work_previous_batch)
        """
        raise NotImplementedError()

    def training_interrupted(self):
        pass

    def train(self):
        if self.execution_stopped:
            log.warning("Can't train: the execution has been stopped.")
            return

        # Set our own signal handler for SIGINT
        signal.signal(signal.SIGINT, self.__signal_handler)
        self.main_pid = os.getpid()

        global_start_time = time.time()
        with SnakeGameExecutor(self.args) as executor:
            batch = self.initial_batch()
            while len(batch) > 0:
                start_time = time.time()
                batch_size = len(batch)

                log.info('Running new batch: %d jobs.', batch_size)
                self.run_counter += batch_size

                results = executor.run_batch(batch)
                batch, qty_work_prev = self.create_batch_from_results(results)

                batch_duration = time.time() - start_time
                log.info('Batch: %d simulations (%g W) in %g sec:'
                         '\n => %g sim/sec'
                         '\n => %g W/sec.',
                         batch_size, qty_work_prev, batch_duration,
                         batch_size / batch_duration,
                         qty_work_prev / batch_duration)

                if self.execution_stopped:
                    self.training_interrupted()
                    break

        training_duration = time.time() - global_start_time
        log.info('Trained with %d simulations in %g sec.',
                 self.run_counter, training_duration)

        # Unregister the signal handler
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def __signal_handler(self, sig, frame):
        if os.getpid() != self.main_pid:
            return

        if sig == signal.SIGINT:
            if not self.execution_stopped:
                log.critical('SIGINT received, stopping...')
                self.execution_stopped = True

                # Unregister the signal handler
                signal.signal(signal.SIGINT, signal.SIG_DFL)
