import random

def generate_4x4_sudoku():
    """
    Generate a random 4x4 sudoku puzzle.
    
    Returns:
        list: A 4x4 2D array representing the sudoku puzzle.
              Empty cells are represented by 0.
    """
    # Create a solved puzzle first
    puzzle = [[0] * 4 for _ in range(4)]
    
    # Fill the puzzle with a valid solution
    if not _fill_puzzle(puzzle):
        return generate_4x4_sudoku()
    
    # Create a copy and remove numbers to create the puzzle
    puzzle_copy = [row[:] for row in puzzle]
    cells = [(i, j) for i in range(4) for j in range(4)]
    random.shuffle(cells)
    
    # Remove approximately half the numbers
    for i, j in cells[:8]:
        puzzle_copy[i][j] = 0
    
    return puzzle_copy

def generate_4x4_pair():
    """
    Generate a random 4x4 sudoku puzzle and its solution.
    
    Returns:
        tuple: (puzzle, solution)
    """
    puzzle = [[0] * 4 for _ in range(4)]
    if not _fill_puzzle(puzzle):
        return generate_4x4_pair()
    
    solution = [row[:] for row in puzzle]
    
    puzzle_copy = [row[:] for row in puzzle]
    cells = [(i, j) for i in range(4) for j in range(4)]
    random.shuffle(cells)
    
    # Remove approximately half the numbers (8 cells)
    for i, j in cells[:8]:
        puzzle_copy[i][j] = 0
        
    return puzzle_copy, solution


def _fill_puzzle(puzzle):
    """
    Recursively fill the puzzle with valid numbers.
    
    Args:
        puzzle: The 4x4 puzzle grid to fill.
    
    Returns:
        bool: True if puzzle was successfully filled, False otherwise.
    """
    # Find next empty cell
    for i in range(4):
        for j in range(4):
            if puzzle[i][j] == 0:
                numbers = list(range(1, 5))
                random.shuffle(numbers)
                
                for num in numbers:
                    if _is_valid(puzzle, i, j, num):
                        puzzle[i][j] = num
                        
                        if _fill_puzzle(puzzle):
                            return True
                        
                        puzzle[i][j] = 0
                
                return False
    
    return True


def _is_valid(puzzle, row, col, num):
    """
    Check if placing num at (row, col) is valid.
    
    Args:
        puzzle: The 4x4 puzzle grid.
        row: Row index.
        col: Column index.
        num: Number to validate.
    
    Returns:
        bool: True if the placement is valid, False otherwise.
    """
    # Check row
    if num in puzzle[row]:
        return False
    
    # Check column
    if num in [puzzle[i][col] for i in range(4)]:
        return False
    
    # Check 2x2 box
    box_row, box_col = 2 * (row // 2), 2 * (col // 2)
    for i in range(box_row, box_row + 2):
        for j in range(box_col, box_col + 2):
            if puzzle[i][j] == num:
                return False
    
    return True