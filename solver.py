import torch
import numpy as np
from model import SudokuGNN
from data_loader import puzzle_to_data

def load_model(model_path="sudoku_gnn.pth", hidden_dim=128):
    """
    Load the trained GNN model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SudokuGNN(hidden_dim=hidden_dim).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    else:
        print(f"Warning: Model file {model_path} not found.")
        return None

import os

def solve_sudoku(puzzle, model_path="sudoku_gnn.pth"):
    """
    Solve a 4x4 Sudoku puzzle using the GNN model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path)
    
    if model is None:
        return None
    
    data = puzzle_to_data(puzzle).to(device)
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
    
    # Reshape prediction from 16 nodes back to 4x4 grid
    solved_grid = pred.cpu().numpy().reshape(4, 4).tolist()
    return solved_grid

if __name__ == "__main__":
    # Example usage
    import new_puzzle
    puzzle = [
        [1, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 3, 0, 0],
        [0, 0, 4, 0]
    ]
    print("Original Puzzle:")
    for row in puzzle:
        print(row)
        
    solved = solve_sudoku(puzzle)
    if solved:
        print("\nGNN Solved:")
        for row in solved:
            print(row)
