# Competitive Snake Game

A two-player competitive Snake game with advanced features built using Python and Pygame.

## Features

- Two-player gameplay with separate controls
- Egg collection with limited lifetime
- Explosion mechanics when eggs expire
- Visual effects for egg consumption
- Score tracking
- Grid-based movement with visible grid lines
- Non-blocking input handling for smooth gameplay

## Controls

### Player 1 (Green Snake)
- W: Move Up
- A: Move Left
- S: Move Down
- D: Move Right

### Player 2 (Blue Snake)
- ↑: Move Up
- ←: Move Left
- ↓: Move Down
- →: Move Right

## Game Rules

1. Each snake starts in the middle of their respective sides
2. Collect eggs to grow longer
3. Eggs have a 7-second lifetime before exploding
4. Explosions create a 5-cell radius danger zone
5. Collision with walls, other snake, self, or explosion zones results in loss
6. If both players lose simultaneously, it's a draw

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Game

```bash
python snake_game.py
```

## Sound Effects

The game includes sound effects for:
- Eating eggs
- Egg explosions

Note: If sound files are not found, the game will run without sound effects.

## Game Mechanics

- Snakes move at a controlled speed for fair gameplay
- Eggs spawn randomly within the grid (away from edges)
- Visual feedback when collecting eggs (snake flashes white)
- Eggs turn red when about to explode
- Explosion zones are shown in orange
- Grid lines help with movement precision