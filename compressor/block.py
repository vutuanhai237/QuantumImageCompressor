
import numpy as np
from .compilation import compilation
class Block:
    def __init__(self, block, i, j):
        self.block: np.ndarray = block
        self.i = i
        self.j = j
        self.thetas = None
        self.cost = None
        self.is_transfered = False
    def __str__(self):
        if self.thetas is None:
            return f'Block at ({self.i}, {self.j}), no thetas'
        return f'Block at ({self.i}, {self.j}), thetas = {self.thetas[0]} ... {self.thetas[-1]}'
    def to_state(self):
        flatten_block = self.block.flatten()
        return flatten_block/np.linalg.norm(flatten_block)

    def find_thetas(self, init_thetas, num_layers):
        costs, thetass = compilation(
            state = self.to_state(), 
            thetas = init_thetas,
            num_layers = num_layers)
        self.cost = np.argmin(costs)
        self.thetas = thetass[np.argmin(costs)]
        