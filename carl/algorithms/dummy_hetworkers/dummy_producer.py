import random
import time

import torch

from carl.algorithms.algorithm import Algorithm
from carl.utils.resources import dump_resource

class DummyProducer(Algorithm):
    def __init__(self, seed: int, janpawel, var) -> None:
        super().__init__()
        self.seed = seed
        self.janpawel = janpawel
        self.var = var

    def run(self) -> None:
        # Generate some tensors
        for _ in range(10):
            x = torch.rand(15, 3, 32, 32)
            y = torch.randint(0, 10, (15,))
            dump_resource({'x': x, 'y': y}, 'experience')
            # random time sleep
            secs = random.randint(3, 10)
            time.sleep(secs)