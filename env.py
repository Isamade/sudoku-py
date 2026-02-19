import torch
import numpy as np
import new_puzzle

class SudokuEnv:
    def __init__(self):
        """
        Custom 4x4 Sudoku environment.
        """
        self.grid = None
        self.solution = None
        self.puzzle_orig = None
        self.reset()

    def reset(self):
        """
        Reset the environment with a new puzzle.
        """
        self.puzzle_orig, self.solution = new_puzzle.generate_4x4_pair()
        self.grid = [row[:] for row in self.puzzle_orig]
        return self._get_obs()

    def _get_obs(self):
        """
        Get the current state observation.
        """
        return torch.tensor(np.array(self.grid).flatten(), dtype=torch.float)

    def step(self, action):
        """
        Take an action in the environment.
        action: tuple (cell_index, value) where cell_index is 0-15 and value is 1-4.
        """
        cell_idx, val = action
        row, col = cell_idx // 4, cell_idx % 4

        # Reward structure
        reward = 0
        done = False
        info = {}

        # Check if the cell is mutable (was zero in original puzzle)
        if self.puzzle_orig[row][col] != 0:
            reward = -1  # Penalty for trying to change a fixed cell
            done = False
        else:
            # Place the value
            self.grid[row][col] = val
            
            # Check against solution
            if val == self.solution[row][col]:
                reward = 1  # Reward for correct placements
            else:
                reward = -0.5 # Penalty for incorrect placement

        # Check if puzzle is full
        if all(cell != 0 for row in self.grid for cell in row):
            done = True
            # Check if solved correctly
            if np.array_equal(self.grid, self.solution):
                reward = 10 # Success reward
                info["success"] = True
            else:
                reward = -2 # Failure penalty
                info["success"] = False

        return self._get_obs(), reward, done, info

    def render(self):
        """
        Simple print render.
        """
        for row in self.grid:
            print(row)
        print("-" * 10)
