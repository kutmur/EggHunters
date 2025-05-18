import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
from enum import Enum
import pygame
import os
import time
import matplotlib.pyplot as plt
from IPython import display

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 15  # Faster for training

class AdvancedSnakeGameAI:
    """Training environment with AI snake and virtual human snake"""
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI Training vs Virtual Human')
        self.clock = pygame.time.Clock()
        self.reset()
        self.font = pygame.font.SysFont('arial', 25)
        self.agent = None  # Will be set by training function

    def reset(self):
        # AI snake
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        # Virtual human snake (on the opposite side)
        self.human_direction = Direction.LEFT
        self.human_head = Point(self.w/4, self.h/2)
        self.human_snake = [self.human_head,
                           Point(self.human_head.x+BLOCK_SIZE, self.human_head.y),
                           Point(self.human_head.x+(2*BLOCK_SIZE), self.human_head.y)]

        self.score = 0
        self.human_score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.human_snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. Get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move AI snake
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. Move virtual human snake - based on a simple strategy
        self._move_human()
        self.human_snake.insert(0, self.human_head)
        
        # 4. Check if game is over
        reward = 0
        game_over = False
        
        # Progressive reward system based on training phase
        # Early training focuses on survival, later on competition
        if hasattr(self, 'agent') and self.agent is not None:
            training_phase = min(1.0, self.agent.n_games / 1000)  # Normalized 0-1 based on games
        else:
            training_phase = 0.5  # Default mid-point if agent reference not available
        
        # Base rewards/penalties
        survival_reward = 0.1  # Small reward for staying alive
        wall_collision_penalty = -10
        self_collision_penalty = -10
        timeout_penalty = -5
        food_reward = 10
        human_got_food_penalty = -5
        
        # Apply training phase adjustments
        competitive_factor = training_phase  # Increases as training progresses
        survival_factor = 1 - (training_phase * 0.5)  # Decreases but never below 0.5
        
        # Basic survival checks
        if self.is_collision(self.head, self.snake[1:]):
            game_over = True
            reward = self_collision_penalty * survival_factor
            return reward, game_over, self.score
        
        if (self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or 
            self.head.y > self.h - BLOCK_SIZE or self.head.y < 0):
            game_over = True
            reward = wall_collision_penalty * survival_factor
            return reward, game_over, self.score
        
        if self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = timeout_penalty
            return reward, game_over, self.score
        
        # Competitive elements (scaled by competitive_factor)
        if self.head in self.human_snake:
            game_over = True
            reward = -20 * competitive_factor  # Penalty for hitting human
            return reward, game_over, self.score
        
        if self.human_head in self.snake:
            reward = 15 * competitive_factor  # Human crashed into AI
            self.human_snake = [self.human_head]  # Human restarts
        
        # Food rewards
        if self.head == self.food:
            self.score += 1
            reward += food_reward
            
            # Bonus for getting food when human is close to it
            human_food_dist = abs(self.human_head.x - self.food.x) + abs(self.human_head.y - self.food.y)
            if human_food_dist < 5 * BLOCK_SIZE:
                reward += 5 * competitive_factor  # Stole food from under human's nose
                
            self._place_food()
        else:
            self.snake.pop()
            reward += survival_reward  # Small reward for surviving
            
        if self.human_head == self.food:
            self.human_score += 1
            reward += human_got_food_penalty * competitive_factor
            self._place_food()
        else:
            self.human_snake.pop()
        
        # Strategic positioning rewards
        if competitive_factor > 0.5:  # Only in later training
            # Reward for cutting off human's path to food
            ai_to_food_x = self.food.x - self.head.x
            ai_to_food_y = self.food.y - self.head.y
            human_to_food_x = self.food.x - self.human_head.x
            human_to_food_y = self.food.y - self.human_head.y
            
            if ((ai_to_food_x * human_to_food_x < 0) and abs(self.head.y - self.human_head.y) < 3 * BLOCK_SIZE) or \
               ((ai_to_food_y * human_to_food_y < 0) and abs(self.head.x - self.human_head.x) < 3 * BLOCK_SIZE):
                reward += 2 * competitive_factor  # Reward for being between human and food
        
        # 6. Update UI
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 7. Return game state
        return reward, game_over, self.score

    def is_collision(self, pt, body_parts):
        # Wall collision
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Self collision
        if pt in body_parts:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw AI snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw human snake
        for pt in self.human_snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw scores
        text1 = self.font.render(f"AI Score: {self.score}", True, BLUE1)
        text2 = self.font.render(f"Human Score: {self.human_score}", True, GREEN1)
        self.display.blit(text1, [0, 0])
        self.display.blit(text2, [self.w-140, 0])
        
        pygame.display.flip()

    def _move(self, action):
        # AI movement - [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        
    def _move_human(self):
        """Simple strategy for virtual human movement."""
        # Goal: Head toward food while avoiding AI
        
        # 1. Identify potential dangers
        dangers = []
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.human_direction)
        
        # Direct collision check with AI
        distance_to_ai = abs(self.human_head.x - self.head.x) + abs(self.human_head.y - self.head.y)
        ai_is_close = distance_to_ai < 3 * BLOCK_SIZE
        
        potential_moves = []
        
        # Evaluate all possible moves
        for i, direction in enumerate(clock_wise):
            # Test move from current position
            test_x = self.human_head.x
            test_y = self.human_head.y
            
            if direction == Direction.RIGHT:
                test_x += BLOCK_SIZE
            elif direction == Direction.LEFT:
                test_x -= BLOCK_SIZE
            elif direction == Direction.DOWN:
                test_y += BLOCK_SIZE
            elif direction == Direction.UP:
                test_y -= BLOCK_SIZE
            
            test_pt = Point(test_x, test_y)
            # Is this move safe?
            is_safe = not (test_pt in self.human_snake[1:] or 
                           test_pt in self.snake or
                           test_x > self.w - BLOCK_SIZE or test_x < 0 or 
                           test_y > self.h - BLOCK_SIZE or test_y < 0)
                           
            if is_safe:
                # Distance to food
                food_distance = abs(test_x - self.food.x) + abs(test_y - self.food.y)
                # Distance to AI
                ai_distance = abs(test_x - self.head.x) + abs(test_y - self.head.y)
                
                score = 0
                # Getting closer to food is good
                score -= food_distance * 0.5
                # Moving away from AI is good (if AI is close)
                if ai_is_close:
                    score += ai_distance * 2
                
                # New: Try to cut off AI's path to food occasionally
                if self.frame_iteration % 15 == 0 and ai_distance < 5 * BLOCK_SIZE:
                    ai_to_food_x = self.food.x - self.head.x
                    ai_to_food_y = self.food.y - self.head.y
                    
                    # If human can intercept AI's path to food
                    if ((ai_to_food_x > 0 and test_x > self.head.x and test_x < self.food.x) or
                        (ai_to_food_x < 0 and test_x < self.head.x and test_x > self.food.x) or
                        (ai_to_food_y > 0 and test_y > self.head.y and test_y < self.food.y) or
                        (ai_to_food_y < 0 and test_y < self.head.y and test_y > self.food.y)):
                        score += 5  # Big bonus for intercepting AI
                
                potential_moves.append((score, direction))
        
        # Prevent going in opposite direction
        opposite_idx = (idx + 2) % 4
        safe_potential_moves = [m for m in potential_moves if m[1] != clock_wise[opposite_idx]]
        
        if not safe_potential_moves and potential_moves:
            # No safe moves left, pick the best score
            potential_moves.sort(key=lambda x: x[0], reverse=True)
            self.human_direction = potential_moves[0][1]
        elif safe_potential_moves:
            # Pick the best score from safe moves
            safe_potential_moves.sort(key=lambda x: x[0], reverse=True)
            self.human_direction = safe_potential_moves[0][1]
        # Else: Continue in current direction
        
        # Apply final movement
        x = self.human_head.x
        y = self.human_head.y
        if self.human_direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.human_direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.human_direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.human_direction == Direction.UP:
            y -= BLOCK_SIZE

        self.human_head = Point(x, y)


class Advanced_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Deeper network with batch normalization
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//2)
        self.linear4 = nn.Linear(hidden_size//2, output_size)
        
    def forward(self, x):
        # Handle single sample case for batch norm
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
            
        # Forward pass with ReLU activations and batch norm
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        
        # Handle single sample case
        if single_sample:
            x = x.squeeze(0)
            
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class PrioritizedMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta    # Importance sampling correction (starts low, anneals to 1)
        self.beta_increment = beta_increment
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        
    def append(self, state, action, reward, next_state, done):
        # New experiences get max priority
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        if self.size < batch_size:
            sample_indices = range(len(self.memory))
        else:
            # Prioritized sampling
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            sample_indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # Importance sampling weights
        weights = (self.size * probabilities[sample_indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)  # Anneal beta
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in sample_indices:
            s, a, r, ns, d = self.memory[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            
        return np.array(states), np.array(actions), np.array(rewards), \
               np.array(next_states), np.array(dones), weights, sample_indices
        
    def update_priorities(self, indices, errors):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + 1e-5  # Small constant to avoid zero


# Customized QTrainer for prioritized replay
class EnhancedQTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Calculate loss and backpropagate
        loss = self.criterion(target, pred)
        loss.backward()
        
        # Update weights
        self.optimizer.step()


class Agent:
    """AI Agent class - enhanced with human opponent awareness"""
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.95  # discount rate
        self.memory = PrioritizedMemory(capacity=100_000)
        self.model = Advanced_QNet(17, 256, 3)  # 17 inputs (including enhanced human info)
        self.trainer = EnhancedQTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_state(self, game):
        """Enhanced state representation with more detailed information."""
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Original danger detection
        danger_straight = (
            (dir_r and (game.is_collision(point_r, game.snake[1:]) or point_r in game.human_snake)) or 
            (dir_l and (game.is_collision(point_l, game.snake[1:]) or point_l in game.human_snake)) or 
            (dir_u and (game.is_collision(point_u, game.snake[1:]) or point_u in game.human_snake)) or 
            (dir_d and (game.is_collision(point_d, game.snake[1:]) or point_d in game.human_snake))
        )
        
        danger_right = (
            (dir_u and (game.is_collision(point_r, game.snake[1:]) or point_r in game.human_snake)) or 
            (dir_d and (game.is_collision(point_l, game.snake[1:]) or point_l in game.human_snake)) or 
            (dir_l and (game.is_collision(point_u, game.snake[1:]) or point_u in game.human_snake)) or 
            (dir_r and (game.is_collision(point_d, game.snake[1:]) or point_d in game.human_snake))
        )
        
        danger_left = (
            (dir_d and (game.is_collision(point_r, game.snake[1:]) or point_r in game.human_snake)) or 
            (dir_u and (game.is_collision(point_l, game.snake[1:]) or point_l in game.human_snake)) or 
            (dir_r and (game.is_collision(point_u, game.snake[1:]) or point_u in game.human_snake)) or 
            (dir_l and (game.is_collision(point_d, game.snake[1:]) or point_d in game.human_snake))
        )
        
        # Move direction
        state = [
            # Danger states
            danger_straight,
            danger_right,
            danger_left,
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y,  # food down
        ]

        # Enhanced human snake information
        human_head = game.human_snake[0]
        
        # Vector to human snake (normalized)
        human_x_diff = (human_head.x - head.x) / game.w
        human_y_diff = (human_head.y - head.y) / game.h
        
        # Distance to human snake head (normalized)
        dist_to_human = (abs(head.x - human_head.x) + abs(head.y - human_head.y)) / (game.w + game.h)
        
        # Is human approaching AI?
        human_approaching = False
        if game.frame_iteration > 1 and len(game.human_snake) > 1:
            prev_dist = (abs(game.snake[0].x - game.human_snake[1].x) + 
                        abs(game.snake[0].y - game.human_snake[1].y)) / (game.w + game.h)
            human_approaching = dist_to_human < prev_dist
        
        # Human food competition 
        human_closer_to_food = (abs(human_head.x - game.food.x) + abs(human_head.y - game.food.y)) < \
                            (abs(head.x - game.food.x) + abs(head.y - game.food.y))
        
        # Add these advanced features to the state
        state.extend([
            human_x_diff, 
            human_y_diff,
            dist_to_human,
            human_approaching,
            human_closer_to_food,
            len(game.snake) > len(game.human_snake),  # AI winning by length
        ])
        
        return np.array(state, dtype=float)  # Changed to float for normalized values

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def train_long_memory(self):
        if self.memory.size < 1000:
            return
            
        states, actions, rewards, next_states, dones, _, _ = self.memory.sample(1000)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Improved epsilon-greedy policy with better decay
        # Fast initial exploration, slower decay over time
        if self.n_games < 50:
            # Very exploratory at start (80% random actions)
            self.epsilon = 80
        elif self.n_games < 200:
            # Linear decay from 80 to 40
            self.epsilon = 80 - ((self.n_games - 50) * 40 / 150)
        else:
            # Slow logarithmic decay afterwards, never below 5%
            try:
                self.epsilon = max(5, 40 - 10 * np.log10(self.n_games - 190))
            except:
                self.epsilon = 5  # Fallback if error in calculation
        
        # Action selection
        final_move = [0, 0, 0]
        if random.random() < self.epsilon / 100:  # Convert to probability
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train_enhanced():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    no_improvement_count = 0
    
    # Create agent with advanced components
    agent = Agent()
    game = AdvancedSnakeGameAI()
    game.agent = agent  # Reference for progressive reward system
    
    # Check and load trained model
    model_folder_path = './model'
    model_path = os.path.join(model_folder_path, 'model_enhanced.pth')
    
    # If trained model exists, load it
    if os.path.exists(model_path):
        try:
            model_state = torch.load(model_path)
            agent.model.load_state_dict(model_state, strict=False)
            print(f"Loaded existing enhanced model: {model_path}")
            print("Continuing training...")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        # Try to load the basic model as starting point if available
        basic_model_path = os.path.join(model_folder_path, 'model.pth')
        if os.path.exists(basic_model_path):
            try:
                model_state = torch.load(basic_model_path)
                
                # Just try loading what we can without enforcing strict matching
                try:
                    agent.model.load_state_dict(model_state, strict=False)
                    print("Successfully loaded parts of basic model as a starting point.")
                except Exception as e:
                    print(f"Partial model loading: {e}")
                    print("Starting with initialized model.")
                
                print("Starting enhanced training...")
            except Exception as e:
                print(f"Error loading basic model: {e}")
                print("Starting from scratch...")
        else:
            print("No existing model found. Starting from scratch...")
    
    # Training monitoring
    moving_avg_score = deque(maxlen=100)
    best_moving_avg = 0
    checkpoint_scores = []
    training_start = time.time()
    
    # Learning rate schedule
    initial_lr = 0.001
    min_lr = 0.00005

    print(f"Training enhanced model with human snake interaction...")
    print(f"Model will be saved as: {model_path}")
    
    try:
        while True:
            # Get old state
            state_old = agent.get_state(game)

            # Get move
            final_move = agent.get_action(state_old)

            # Perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # Train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # Remember for experience replay
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Train long memory and reset game
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                # Tracking scores for monitoring
                if score > record:
                    record = score
                    # Save model
                    if not os.path.exists(model_folder_path):
                        os.makedirs(model_folder_path)
                    torch.save(agent.model.state_dict(), model_path)
                    print(f'New record! Model saved to {model_path}')
                
                moving_avg_score.append(score)
                current_avg = sum(moving_avg_score) / len(moving_avg_score)
                
                # Save checkpoints every 500 games
                if agent.n_games % 500 == 0:
                    checkpoint_path = os.path.join(model_folder_path, f'model_checkpoint_{agent.n_games}.pth')
                    torch.save(agent.model.state_dict(), checkpoint_path)
                    checkpoint_scores.append((agent.n_games, current_avg))
                
                # Learning rate scheduling - reduce over time
                if agent.n_games % 200 == 0 and agent.n_games > 0:
                    lr = max(min_lr, initial_lr * (0.95 ** (agent.n_games // 200)))
                    agent.trainer.lr = lr
                    print(f"Learning rate adjusted to {lr:.6f}")
                
                # Early stopping logic
                if len(moving_avg_score) == moving_avg_score.maxlen:  # Only after we have enough games
                    if current_avg > best_moving_avg:
                        best_moving_avg = current_avg
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    # If no improvement for 2000 games, stop training
                    if no_improvement_count >= 2000:
                        print(f"Early stopping triggered after {agent.n_games} games with no improvement")
                        raise KeyboardInterrupt  # Use the same exit method

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

                # Print time and progress
                if agent.n_games % 100 == 0:
                    mins_elapsed = (time.time() - training_start) / 60
                    print(f"Game {agent.n_games}, Score {score}, Avg Score: {current_avg:.1f}, Record: {record}")
                    print(f"Training for {mins_elapsed:.1f} minutes, Epsilon: {agent.epsilon:.1f}%")
                    
                    # Print performance trend
                    if len(checkpoint_scores) >= 2:
                        last_checkpoint, last_avg = checkpoint_scores[-1]
                        prev_checkpoint, prev_avg = checkpoint_scores[-2]
                        improvement = last_avg - prev_avg
                        print(f"Performance trend: {'+' if improvement > 0 else ''}{improvement:.2f} " + 
                              f"avg score from game {prev_checkpoint} to {last_checkpoint}")

    except KeyboardInterrupt:
        print("Training interrupted")
        # Save final model state
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        # Save final model
        final_path = os.path.join(model_folder_path, 'model_enhanced_final.pth')
        torch.save(agent.model.state_dict(), final_path)
        print(f"Final model saved to {final_path}")
        
        # Plot training history
        if len(checkpoint_scores) > 1:
            plt.figure(figsize=(12, 6))
            plt.plot([x[0] for x in checkpoint_scores], [x[1] for x in checkpoint_scores], 'b-')
            plt.title('Training Progress')
            plt.xlabel('Games')
            plt.ylabel('Average Score (100 games)')
            plt.grid(True)
            plt.savefig(os.path.join(model_folder_path, 'training_history.png'))
            print(f"Training history plotted and saved")


if __name__ == '__main__':
    try:
        print("Starting enhanced AI training...")
        train_enhanced()
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        traceback.print_exc()
