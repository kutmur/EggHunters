import pygame
import random
import sys
import time
from pygame import mixer

# Initialize Pygame and mixer
pygame.init()
mixer.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE
EGG_LIFETIME = 7  # seconds
EXPLOSION_RADIUS = 5

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

class Explosion:
    def __init__(self, position):
        self.position = position
        self.radius = EXPLOSION_RADIUS
        self.active = True
        self.creation_time = time.time()
        self.duration = 2  # seconds

    def is_active(self):
        return time.time() - self.creation_time < self.duration

    def get_danger_zone(self):
        x, y = self.position
        danger_zone = set()
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                if dx*dx + dy*dy <= self.radius*self.radius:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT:
                        danger_zone.add((new_x, new_y))
        return danger_zone

class Snake:
    def __init__(self, color, start_pos, controls):
        self.color = color
        self.body = [start_pos]
        self.direction = (1, 0)  # Initial direction (right)
        self.controls = controls
        self.score = 1
        self.flash_time = 0
        self.is_flashing = False

    def move(self):
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.body.insert(0, new_head)
        self.body.pop()

    def grow(self):
        self.body.append(self.body[-1])
        self.score += 1
        self.is_flashing = True
        self.flash_time = time.time()

    def check_collision(self, other_snake=None, explosions=None):
        head = self.body[0]
        
        # Check wall collision
        if (head[0] < 0 or head[0] >= GRID_WIDTH or 
            head[1] < 0 or head[1] >= GRID_HEIGHT):
            return True

        # Check self collision
        if head in self.body[1:]:
            return True

        # Check collision with other snake
        if other_snake:
            if head in other_snake.body:
                return True

        # Check explosion collision
        if explosions:
            for explosion in explosions:
                if explosion.active and head in explosion.get_danger_zone():
                    return True

        return False

    def get_color(self):
        if self.is_flashing and time.time() - self.flash_time < 0.2:
            return WHITE
        return self.color

class Egg:
    def __init__(self):
        self.position = self.generate_position()
        self.creation_time = time.time()
        self.is_exploding = False

    def generate_position(self):
        # Generate position away from edges
        x = random.randint(2, GRID_WIDTH - 3)
        y = random.randint(2, GRID_HEIGHT - 3)
        return (x, y)

    def respawn(self, snake1, snake2):
        while True:
            new_pos = self.generate_position()
            if (new_pos not in snake1.body and 
                new_pos not in snake2.body):
                self.position = new_pos
                self.creation_time = time.time()
                self.is_exploding = False
                break

    def should_explode(self):
        return time.time() - self.creation_time >= EGG_LIFETIME

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Competitive Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Load sounds
        try:
            self.eat_sound = mixer.Sound('eat.wav')
            self.explosion_sound = mixer.Sound('explosion.wav')
        except:
            print("Sound files not found. Game will run without sound.")
            self.eat_sound = None
            self.explosion_sound = None
        
        # Initialize snakes with positions away from edges
        self.snake1 = Snake(GREEN, (GRID_WIDTH//4, GRID_HEIGHT//2), {
            pygame.K_w: (0, -1),  # Up
            pygame.K_s: (0, 1),   # Down
            pygame.K_a: (-1, 0),  # Left
            pygame.K_d: (1, 0)    # Right
        })
        
        self.snake2 = Snake(BLUE, (3*GRID_WIDTH//4, GRID_HEIGHT//2), {
            pygame.K_UP: (0, -1),    # Up
            pygame.K_DOWN: (0, 1),   # Down
            pygame.K_LEFT: (-1, 0),  # Left
            pygame.K_RIGHT: (1, 0)   # Right
        })
        
        self.egg = Egg()
        self.explosions = []
        self.game_over = False
        self.winner = None
        self.last_update = time.time()
        self.update_interval = 0.15  # Slower speed for better gameplay

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Non-blocking input handling
        keys = pygame.key.get_pressed()
        
        # Player 1 controls
        for key, direction in self.snake1.controls.items():
            if keys[key]:
                # Prevent 180-degree turns
                if (direction[0] != -self.snake1.direction[0] or 
                    direction[1] != -self.snake1.direction[1]):
                    self.snake1.direction = direction
                break
        
        # Player 2 controls
        for key, direction in self.snake2.controls.items():
            if keys[key]:
                # Prevent 180-degree turns
                if (direction[0] != -self.snake2.direction[0] or 
                    direction[1] != -self.snake2.direction[1]):
                    self.snake2.direction = direction
                break

    def update(self):
        if self.game_over:
            return

        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        self.last_update = current_time

        # Move snakes
        self.snake1.move()
        self.snake2.move()

        # Check egg lifetime
        if self.egg.should_explode() and not self.egg.is_exploding:
            self.egg.is_exploding = True
            self.explosions.append(Explosion(self.egg.position))
            if self.explosion_sound:
                self.explosion_sound.play()
            self.egg.respawn(self.snake1, self.snake2)

        # Remove expired explosions
        self.explosions = [exp for exp in self.explosions if exp.is_active()]

        # Check collisions
        snake1_collision = self.snake1.check_collision(self.snake2, self.explosions)
        snake2_collision = self.snake2.check_collision(self.snake1, self.explosions)

        if snake1_collision and snake2_collision:
            self.game_over = True
            self.winner = "Draw"
        elif snake1_collision:
            self.game_over = True
            self.winner = "Player 2"
        elif snake2_collision:
            self.game_over = True
            self.winner = "Player 1"

        # Check egg collision
        if self.snake1.body[0] == self.egg.position:
            self.snake1.grow()
            if self.eat_sound:
                self.eat_sound.play()
            self.egg.respawn(self.snake1, self.snake2)
        elif self.snake2.body[0] == self.egg.position:
            self.snake2.grow()
            if self.eat_sound:
                self.eat_sound.play()
            self.egg.respawn(self.snake1, self.snake2)

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw grid lines
        for x in range(0, WINDOW_WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (WINDOW_WIDTH, y))

        # Draw explosions
        for explosion in self.explosions:
            for pos in explosion.get_danger_zone():
                pygame.draw.rect(self.screen, ORANGE,
                               (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE,
                                GRID_SIZE - 2, GRID_SIZE - 2))

        # Draw snakes
        for segment in self.snake1.body:
            pygame.draw.rect(self.screen, self.snake1.get_color(),
                           (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                            GRID_SIZE - 2, GRID_SIZE - 2))
        
        for segment in self.snake2.body:
            pygame.draw.rect(self.screen, self.snake2.get_color(),
                           (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                            GRID_SIZE - 2, GRID_SIZE - 2))

        # Draw egg with lifetime indicator
        if not self.egg.is_exploding:
            egg_color = YELLOW
            if self.egg.should_explode():
                egg_color = RED
            pygame.draw.rect(self.screen, egg_color,
                           (self.egg.position[0] * GRID_SIZE,
                            self.egg.position[1] * GRID_SIZE,
                            GRID_SIZE - 2, GRID_SIZE - 2))

        # Draw scores
        score1_text = self.font.render(f"P1: {self.snake1.score}", True, GREEN)
        score2_text = self.font.render(f"P2: {self.snake2.score}", True, BLUE)
        self.screen.blit(score1_text, (10, 10))
        self.screen.blit(score2_text, (WINDOW_WIDTH - 100, 10))

        # Draw game over screen
        if self.game_over:
            if self.winner == "Draw":
                text = "Game Over - Draw!"
            else:
                text = f"Game Over - {self.winner} Wins!"
            game_over_text = self.font.render(text, True, WHITE)
            text_rect = game_over_text.get_rect(center=(WINDOW_WIDTH/2, WINDOW_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

        pygame.display.flip()

    def run(self):
        while True:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60)  # Cap at 60 FPS for smooth animations

if __name__ == "__main__":
    game = Game()
    game.run() 