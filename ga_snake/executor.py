import os
from concurrent.futures.process import ProcessPoolExecutor

from ga_snake.client import run_simulation
from ga_snake.snake import Snake


class Executor(object):
    def __init__(self, args):
        self.hpv = args.host, args.port, args.venue
        self.executor = ProcessPoolExecutor(max_workers=os.cpu_count())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)
        return False

    def run_batch(self, batch):
        results = []
        params = [(*self.hpv, Snake(data)) for data in batch]
        for result in self.executor.map(run_simulation, params):
            results.append(result)
        return results
