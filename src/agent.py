import torch
import random
import numpy as np
from collections import deque
from src.core import Direction, Point, BLOCK_SIZE
from src.model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness control
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)  # 11 inputs, 256 hidden, 3 outputs (straight, right, left)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        """
        Get the current state of the game for the AI to process
        Returns array with 11 values representing the game state
        """
        head = game.ai_snake[0]
        
        # Check points around the head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Current direction
        dir_l = game.ai_direction == Direction.LEFT
        dir_r = game.ai_direction == Direction.RIGHT
        dir_u = game.ai_direction == Direction.UP
        dir_d = game.ai_direction == Direction.DOWN

        # Create state array
        state = [
            # Danger straight
            (dir_r and game.is_collision_ai(point_r)) or 
            (dir_l and game.is_collision_ai(point_l)) or 
            (dir_u and game.is_collision_ai(point_u)) or 
            (dir_d and game.is_collision_ai(point_d)),

            # Danger right
            (dir_u and game.is_collision_ai(point_r)) or 
            (dir_d and game.is_collision_ai(point_l)) or 
            (dir_l and game.is_collision_ai(point_u)) or 
            (dir_r and game.is_collision_ai(point_d)),

            # Danger left
            (dir_d and game.is_collision_ai(point_r)) or 
            (dir_u and game.is_collision_ai(point_l)) or 
            (dir_r and game.is_collision_ai(point_u)) or 
            (dir_l and game.is_collision_ai(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y   # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Train on a batch from memory (experience replay)"""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train for a single step"""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Decide the next move (epsilon-greedy approach)"""
        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]  # [straight, right, left]
        
        # More exploration at the beginning, less over time
        if random.randint(0, 200) < self.epsilon:
            # Random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Model-predicted move
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
