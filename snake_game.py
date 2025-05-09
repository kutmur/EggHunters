import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class Snake:
    def __init__(self, color, start_pos, controls):
        self.color = color
        self.body = [start_pos]
        self.direction = (1, 0)  # Initial direction (right)
        self.controls = controls
        self.score = 1

    def move(self):
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.body.insert(0, new_head)
        self.body.pop()

    def grow(self):
        self.body.append(self.body[-1])
        self.score += 1

    def check_collision(self, other_snake=None):
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

        return False

class Egg:
    def __init__(self):
        self.position = self.generate_position()

    def generate_position(self):
        x = random.randint(0, GRID_WIDTH - 1)
        y = random.randint(0, GRID_HEIGHT - 1)
        return (x, y)

    def respawn(self, snake1, snake2):
        while True:
            new_pos = self.generate_position()
            if (new_pos not in snake1.body and 
                new_pos not in snake2.body):
                self.position = new_pos
                break

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Competitive Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize snakes
        self.snake1 = Snake(GREEN, (5, GRID_HEIGHT // 2), {
            pygame.K_w: (0, -1),  # Up
            pygame.K_s: (0, 1),   # Down
            pygame.K_a: (-1, 0),  # Left
            pygame.K_d: (1, 0)    # Right
        })
        
        self.snake2 = Snake(BLUE, (GRID_WIDTH - 6, GRID_HEIGHT // 2), {
            pygame.K_UP: (0, -1),    # Up
            pygame.K_DOWN: (0, 1),   # Down
            pygame.K_LEFT: (-1, 0),  # Left
            pygame.K_RIGHT: (1, 0)   # Right
        })
        
        self.egg = Egg()
        self.game_over = False
        self.winner = None

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                # Player 1 controls
                if event.key in self.snake1.controls:
                    new_direction = self.snake1.controls[event.key]
                    # Prevent 180-degree turns
                    if (new_direction[0] != -self.snake1.direction[0] or 
                        new_direction[1] != -self.snake1.direction[1]):
                        self.snake1.direction = new_direction
                
                # Player 2 controls
                if event.key in self.snake2.controls:
                    new_direction = self.snake2.controls[event.key]
                    # Prevent 180-degree turns
                    if (new_direction[0] != -self.snake2.direction[0] or 
                        new_direction[1] != -self.snake2.direction[1]):
                        self.snake2.direction = new_direction

    def update(self):
        if self.game_over:
            return

        # Move snakes
        self.snake1.move()
        self.snake2.move()

        # Check collisions
        snake1_collision = self.snake1.check_collision(self.snake2)
        snake2_collision = self.snake2.check_collision(self.snake1)

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
            self.egg.respawn(self.snake1, self.snake2)
        elif self.snake2.body[0] == self.egg.position:
            self.snake2.grow()
            self.egg.respawn(self.snake1, self.snake2)

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw grid lines
        for x in range(0, WINDOW_WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (WINDOW_WIDTH, y))

        # Draw snakes
        for segment in self.snake1.body:
            pygame.draw.rect(self.screen, self.snake1.color,
                           (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                            GRID_SIZE - 2, GRID_SIZE - 2))
        
        for segment in self.snake2.body:
            pygame.draw.rect(self.screen, self.snake2.color,
                           (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                            GRID_SIZE - 2, GRID_SIZE - 2))

        # Draw egg
        pygame.draw.rect(self.screen, YELLOW,
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
            self.clock.tick(10)  # Control game speed

if __name__ == "__main__":
    game = Game()
    game.run() 