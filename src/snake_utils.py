import pygame
import os
from enum import Enum
from collections import namedtuple

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Constants
BLOCK_SIZE = 20
SPEED_CLASSIC = 15
SPEED_PVP = 15
SPEED_AI = 20

# Define Point namedtuple
Point = namedtuple('Point', 'x, y')

# Define Direction enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)
BLACK = (0, 0, 0)

# Sounds
def load_sounds():
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

# Function to check collision with walls or snake body
def is_collision(point, snake_body, w, h):
    # Hits boundary
    if point.x > w - BLOCK_SIZE or point.x < 0 or point.y > h - BLOCK_SIZE or point.y < 0:
        return True
    # Hits itself
    if point in snake_body[1:]:
        return True
    return False

# Draw a snake
def draw_snake(display, snake_body, color1, color2):
    for pt in snake_body:
        pygame.draw.rect(display, color1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(display, color2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

# Draw food
def draw_food(display, food):
    pygame.draw.rect(display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
