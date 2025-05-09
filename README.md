# Competitive Snake Game

A two-player competitive Snake game built with Python and Pygame.

## Requirements

- Python 3.x
- Pygame 2.5.2

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## How to Play

Run the game:
```bash
python snake_game.py
```

### Controls

Player 1 (Green Snake):
- W: Move Up
- S: Move Down
- A: Move Left
- D: Move Right

Player 2 (Blue Snake):
- ↑: Move Up
- ↓: Move Down
- ←: Move Left
- →: Move Right

### Game Rules

- Each snake grows by 1 unit when it eats an egg (yellow square)
- A snake loses if it:
  - Hits the wall
  - Collides with its own body
  - Collides with the other snake's body
- If both snakes crash at the same time, it's a draw
- The current score (snake length) is displayed for each player
- The game ends when one player wins or there's a draw