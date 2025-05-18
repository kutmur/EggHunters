# Snake Game Collection

A modern collection of Snake game variants built with Pygame, including single player, two-player PvP, and human vs AI modes.

## Game Modes

1. **Classic Single-Player Mode**: Traditional snake game with arrow keys control
2. **Player vs Player Mode**: Two players controlled with WASD (Player 1) and Arrow Keys (Player 2)
3. **Human vs AI Mode**: Play against a Deep Q-Network (DQN) reinforcement learning agent

## Installation

Ensure you have Python 3.6+ and pip installed. Then run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

## How to Play

### Using the Game Launcher
```bash
python snake_game_launcher.py
```
- Launch the graphical menu to select which game mode to play
- Click on your desired game mode to start playing
- ESC key exits any game and returns to the launcher

### Classic Snake (Single Player)
```bash
python -m src.snake_classic
```
- Use Arrow Keys to control the snake
- Eat food to grow and increase your score
- Avoid collisions with walls and your own body
- Press SPACE to restart after game over
- Press ESC to quit

### Snake PvP (Player vs Player)
```bash
python -m src.snake_pvp
```
- Player 1 (Green): Use W, A, S, D keys
- Player 2 (Blue): Use Arrow Keys
- Collect food to grow and score points
- Avoid hitting walls, yourself, or the other player
- The player with the highest score when both die wins
- Press SPACE to restart after game over
- Press ESC to quit

### Snake Human vs AI
```bash
python -m src.snake_human_vs_ai
```
- Human (Green): Use Arrow Keys
- AI (Blue): Controlled by a trained neural network
- Compete against the AI to get a higher score
- AI uses a pre-trained model from the `models` directory
- Press SPACE to restart after game over
- Press ESC to quit

## Sound Effects

The game includes sound effects for:
- Eating food
- Game over
- Winning

Note: If sound files are not found, the game will run without sound effects.

## Project Structure

```
project-root/
├── src/
│   ├── snake_classic.py   # Single-player Snake
│   ├── snake_pvp.py       # Two-player PvP mode
│   ├── snake_human_vs_ai.py  # Human vs AI mode
│   ├── core.py           # Shared game logic
│   ├── agent.py          # AI agent implementation
│   ├── model.py          # Neural network model
│   └── helper.py         # Helper functions
├── assets/
│   ├── eat.wav           # Sound effect for eating food
│   ├── win.wav           # Sound effect for winning
│   └── game_over.wav     # Sound effect for game over
├── models/
│   └── model.pth         # Trained AI model
├── snake_game_launcher.py  # Main launcher for all game modes
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Game Mechanics

- Snakes move at a controlled speed for fair gameplay
- Food spawns randomly within the grid (away from snake bodies)
- Each game mode has its own scoring system and win conditions
- AI uses a reinforcement learning model to make smart decisions
- Eggs turn red when about to explode
- Explosion zones are shown in orange
- Grid lines help with movement precision