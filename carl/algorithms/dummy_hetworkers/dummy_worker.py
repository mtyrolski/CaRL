import random
import time

from loguru import logger

from carl.algorithms.algorithm import Algorithm
from carl.utils.resources import read_resource_and_delete


class DummyReceiver(Algorithm):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.total_received = 0
        self.seed = seed

    def run(self) -> None:
        # Receive some tensors in loop and log current ammount
        for _ in range(20):
            experiences = read_resource_and_delete('experience')
            logger.info(f'Received {len(experiences)} experiences')
            self.total_received += len(experiences)
            logger.info(f'Total received: {self.total_received + len(experiences)}')
            # random time sleep
            secs = random.randint(5, 8)
            time.sleep(secs)