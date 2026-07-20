from collections import defaultdict

import numpy as np
import torch


class Averager:
    def __init__(self):
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.total = {}
        self.counter = {}

    def update(self, stats):
        for key, value in stats.items():
            if key not in self.total:
                if isinstance(value, torch.Tensor):
                    self.total[key] = value.sum()
                    self.counter[key] = value.numel()
                elif isinstance(value, np.ndarray):
                    self.total[key] = value.sum()
                    self.counter[key] = value.size
                else:
                    self.total[key] = value
                    self.counter[key] = 1
            else:
                if isinstance(value, torch.Tensor):
                    self.total[key] = self.total[key] + value.sum()
                    self.counter[key] = self.counter[key] + value.numel()
                elif isinstance(value, np.ndarray):
                    self.total[key] = self.total[key] + value.sum()
                    self.counter[key] = self.counter[key] + value.size
                else:
                    self.total[key] = self.total[key] + value
                    self.counter[key] = self.counter[key] + 1

    def average(self):
        averaged_stats = {
            key: (tot / self.counter[key]).item() if isinstance(tot, torch.Tensor) else tot / self.counter[key] for key, tot in self.total.items()
        }
        self.reset()

        return averaged_stats
