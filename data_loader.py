import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import new_puzzle

def get_sudoku_edges():
    """
    Construct edge indices for a 4x4 Sudoku graph.
    Edges exist between cells in the same row, column, or 2x2 box.
    """
    edge_index = []
    
    for i in range(16):
        r1, c1 = i // 4, i % 4
        b1 = (r1 // 2) * 2 + (c1 // 2)
        
        for j in range(16):
            if i == j:
                continue
            
            r2, c2 = j // 4, j % 4
            b2 = (r2 // 2) * 2 + (c2 // 2)
            
            # Same row, column, or box
            if r1 == r2 or c1 == c2 or b1 == b2:
                edge_index.append([i, j])
                
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def puzzle_to_data(puzzle, solution=None):
    """
    Convert a 4x4 puzzle (and optional solution) to a PyTorch Geometric Data object.
    """
    # Flatten the puzzle to 16 nodes
    x = torch.tensor(np.array(puzzle).flatten(), dtype=torch.float).view(-1, 1)
    
    edge_index = get_sudoku_edges()
    
    if solution is not None:
        y = torch.tensor(np.array(solution).flatten(), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)
    else:
        return Data(x=x, edge_index=edge_index)

class SudokuDataset(Dataset):
    def __init__(self, num_samples=1000):
        super(SudokuDataset, self).__init__()
        self.num_samples = num_samples
        self.edge_index = get_sudoku_edges()
        
    def len(self):
        return self.num_samples
    
    def get(self, idx):
        puzzle, solution = new_puzzle.generate_4x4_pair()
        return puzzle_to_data(puzzle, solution)
