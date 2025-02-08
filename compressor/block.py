
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
        self.num_steps = 0
        self.min_steps = 0
    def __str__(self):
        if self.thetas is None:
            return f'Block at ({self.i}, {self.j}), no thetas'
        return f'Block at ({self.i}, {self.j}), thetas = {self.thetas[0]} ... {self.thetas[-1]}'
    def to_state(self):
        state = self.block.flatten()
        if state.shape[0] != 2**int(np.log2(state.shape[0])):
            new_size = 2**int(np.ceil(np.log2(state.shape[0])))
            padded_state = np.zeros(new_size, dtype=state.dtype)
            padded_state[:state.shape[0]] = state
            state = padded_state
        return state/np.linalg.norm(state)

    def find_thetas(self, init_thetas, num_layers):
        costs, thetass, num_steps = compilation(
            state = self.to_state(), 
            thetas = init_thetas,
            num_layers = num_layers)
        self.num_steps = num_steps
        self.cost = np.min(costs)
        self.min_steps = np.argmin(costs)
        self.thetas = thetass[np.argmin(costs)]
        