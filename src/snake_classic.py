import pygame
import random
import sys
from src.core import SnakeGame, Direction, Point, BLOCK_SIZE, BLACK, WHITE
from src.core import BLUE1, BLUE2, SPEED_CLASSIC, draw_snake, draw_food, is_collision

class SnakeClassic(SnakeGame):
    def __init__(self, w=640, h=480):
        super().__init__(w, h, "Snake Classic")
        self.reset()
    
    def reset(self):
        """Reset the game state"""
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head, 
            Point(self.head.x-BLOCK_SIZE, self.head.y),
            Point(self.head.x-(2*BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = self.place_food([self.snake])
        self.game_over = False
        self.game_over = False
    
    def play_step(self):
        """Process one frame of the game"""
        # 1. Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        # 2. Move snake
        new_head, self.direction = self.move_snake(self.direction, self.head)
        self.head = new_head
        self.snake.insert(0, self.head)
        
        # 3. Check collision
        if is_collision(self.head, self.snake, self.w, self.h):
            if 'game_over' in self.sounds:
                self.sounds['game_over'].play()
            self.game_over = True
            return
        
        # 4. Check food
        if self.head == self.food:
            self.score += 1
            if 'eat' in self.sounds:
                self.sounds['eat'].play()
            self.food = self.place_food([self.snake])
        else:
            self.snake.pop()
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED_CLASSIC)
    
    def _update_ui(self):
        """Render the game display"""
        self.display.fill(BLACK)
        
        # Draw snake and food
        draw_snake(self.display, self.snake, BLUE1, BLUE2)
        draw_food(self.display, self.food)
        
        # Draw score
        self.show_score(self.score)
        
        pygame.display.flip()
    
def main():
    game = SnakeClassic()
    
    while True:
        if game.game_over:
            game.show_game_over()
            # Wait for key press to restart or exit
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            game.reset()
                            waiting = False
                        elif event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
        else:
            game.play_step()

if __name__ == "__main__":
    main()
