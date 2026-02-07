from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import new_puzzle

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/new-game")
def get_new_game():
    """
    Returns a new 4x4 Sudoku puzzle.
    """
    puzzle = new_puzzle.generate_4x4_sudoku()
    return {"puzzle": puzzle}

