import numpy as np
from .constant import tau
from .utils import divide_image_into_blocks, create_random_indices
from .block import Block
from .converter import thetas_x_to_thetas_y


class Image:
    def __init__(self, img, k):
        self.img = img
        self.k = k
        blocks = divide_image_into_blocks(img, k)
        self.blocks = np.empty(shape=(len(blocks), len(blocks[0])), dtype=Block)
        for i in range(len(blocks)):
            for j in range(len(blocks[0])):
                self.blocks[i][j] = Block(blocks[i][j], i, j)
        self.num_blocks = self.blocks.shape[0] * self.blocks.shape[1]
        return
    def find_thetas_naive(self, num_layers):
        for i in range(len(self.blocks)):
            for j in range(len(self.blocks[0])):
                self.blocks[i][j].find_thetas(None, num_layers)
        return
    def find_thetas(self, num_layers):
        indices = create_random_indices(len(self.blocks), len(self.blocks[0]))
        for i, j in indices:
            self.find_thetas_at_ij(i, j, num_layers)
        return

    def find_thetas_at_ij(
        self, i, j, num_layers, is_optimize_even_thetas_not_none=True
    ):
        # Consider the block at (i,j) and its neighbors
        # If any neighbor has thetas, approximating thetas of the block at (i,j)
        # by thetas of the neighbor
        neighbors = self.get_neighbor_blocks(i, j)
        current_cost_func_at_y_by_thetas_y = 1
        init_thetas = None
        if len(neighbors) != 0:
            for neighbor in neighbors:
                y = self.blocks[i][j].to_state()
                thetas_y, cost_func_at_y_by_thetas_y = thetas_x_to_thetas_y(
                    y, neighbor.thetas, num_layers
                )
                if cost_func_at_y_by_thetas_y < current_cost_func_at_y_by_thetas_y:
                    current_cost_func_at_y_by_thetas_y = cost_func_at_y_by_thetas_y
                    init_thetas = thetas_y
        if  init_thetas is not None:
            print(f"Block at ({i},{j}) is transfered")
            self.blocks[i][j].thetas = init_thetas
            self.blocks[i][j].cost = current_cost_func_at_y_by_thetas_y
            self.blocks[i][j].is_transfered = True
            if current_cost_func_at_y_by_thetas_y > tau:
                self.blocks[i][j].find_thetas(init_thetas, num_layers)
            else:
                print(f"Cost is less than tau, Block at ({i},{j}) is no need to finding thetas")
        else:
            print(f"Block at ({i},{j}) is finding thetas by itself")
            self.blocks[i][j].find_thetas(init_thetas, num_layers)
        return

    def get_neighbor_blocks(self, i, j):
        neighbors = []
        for x in range(max(0, i - 1), min(len(self.blocks), i + 2)):
            for y in range(max(0, j - 1), min(len(self.blocks[0]), j + 2)):
                if x == i and y == j:
                    continue
                if self.blocks[x][y].thetas is not None:
                    neighbors.append(self.blocks[x][y])
        return neighbors
    def get_min_stepss(self):
        min_stepss = []
        for i in range(self.blocks.shape[0]):
            for j in range(self.blocks.shape[1]):
                min_stepss.append(self.blocks[i][j].min_steps)
        return np.array(min_stepss)
    def get_costs(self):
        costs = []
        for i in range(self.blocks.shape[0]):
            for j in range(self.blocks.shape[1]):
                costs.append(self.blocks[i][j].cost)
        return np.array(costs)
    def get_num_transfered(self):
        num_transfered = 0
        for i in range(self.blocks.shape[0]):
            for j in range(self.blocks.shape[1]):
                if self.blocks[i][j].is_transfered:
                    num_transfered += 1
        return num_transfered
