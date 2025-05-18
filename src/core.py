import pygame
import random
import os
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Define Point namedtuple
Point = namedtuple('Point', 'x, y')

# Define Direction enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Constants
BLOCK_SIZE = 20
SPEED_CLASSIC = 15
SPEED_PVP = 15
SPEED_AI = 20

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)
BLACK = (0, 0, 0)

def load_sounds():
    """Load all sound effects"""
    sounds = {}
    sound_files = {
        'eat': 'eat.wav',
        'game_over': 'game_over.wav',
        'win': 'win.wav'
    }
    
    for name, file in sound_files.items():
        try:
            sound_path = os.path.join('assets', file)
            sounds[name] = pygame.mixer.Sound(sound_path)
        except pygame.error:
            print(f"Warning: Could not load sound {sound_path}")
    
    return sounds

def is_collision(point, snake, w, h):
    """Check if there's a collision with boundaries or snake body"""
    # Hits boundary
    if point.x > w - BLOCK_SIZE or point.x < 0 or point.y > h - BLOCK_SIZE or point.y < 0:
        return True
    # Hits snake body (except head)
    if point in snake[1:]:
        return True
    return False

def draw_snake(display, snake, color1, color2):
    """Draw a snake on the display"""
    for pt in snake:
        pygame.draw.rect(display, color1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(display, color2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

def draw_food(display, food):
    """Draw food on the display"""
    pygame.draw.rect(display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))

class SnakeGame:
    """Base class for all snake game variants"""
    def __init__(self, w=640, h=480, caption="Snake Game"):
        self.w = w
        self.h = h
        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 25)
        self.sounds = load_sounds()
    
    def place_food(self, snake_list=None):
        """Place food at random position, avoiding snakes"""
        if snake_list is None:
            snake_list = [self.snake]
        
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        food = Point(x, y)
        
        # Check if food collides with any snake
        for snake in snake_list:
            if food in snake:
                return self.place_food(snake_list)  # Try again
        return food
    
    def move_snake(self, direction, head, turn_left=False, turn_right=False):
        """Calculate new head position based on direction and turns"""
        x = head.x
        y = head.y
        
        # Calculate new direction if turning
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(direction)
        
        if turn_right:
            new_dir = clock_wise[(idx + 1) % 4]
        elif turn_left:
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = direction
            
        # Move in the new direction
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE
            
        return Point(x, y), new_dir
    
    def show_score(self, score, x=0, y=0):
        """Display score on screen"""
        text = self.font.render(f"Score: {score}", True, WHITE)
        self.display.blit(text, [x, y])
    
    def show_game_over(self, winner=None):
        """Show game over screen with optional winner"""
        self.display.fill(BLACK)
        
        # Game Over text
        text = self.font.render("GAME OVER", True, WHITE)
        self.display.blit(text, [self.w/2 - text.get_width()/2, self.h/3])
        
        # Winner text (if applicable)
        if winner:
            win_text = self.font.render(f"{winner} WINS!", True, WHITE)
            self.display.blit(win_text, [self.w/2 - win_text.get_width()/2, self.h/2])
        
        # Restart instructions
        restart_text = self.font.render("Press SPACE to restart or ESC to exit", True, WHITE)
        self.display.blit(restart_text, [self.w/2 - restart_text.get_width()/2, 2*self.h/3])
        
        pygame.display.flip()
