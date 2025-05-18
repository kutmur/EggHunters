import pygame
import random
import sys
from src.core import SnakeGame, Direction, Point, BLOCK_SIZE
from src.core import BLACK, WHITE, BLUE1, BLUE2, GREEN1, GREEN2, SPEED_PVP
from src.core import draw_snake, draw_food, is_collision

class SnakePvP(SnakeGame):
    def __init__(self, w=640, h=480):
        super().__init__(w, h, "Snake PvP - WASD vs Arrow Keys")
        self.reset()
    
    def reset(self):
        """Reset the game state for both players"""
        # Player 1 (WASD) - Left side of screen
        self.p1_direction = Direction.RIGHT
        self.p1_head = Point(self.w/4, self.h/2)
        self.p1_snake = [self.p1_head,
                        Point(self.p1_head.x-BLOCK_SIZE, self.p1_head.y),
                        Point(self.p1_head.x-(2*BLOCK_SIZE), self.p1_head.y)]
        self.p1_score = 0
        self.p1_alive = True
        
        # Player 2 (Arrow Keys) - Right side of screen
        self.p2_direction = Direction.LEFT
        self.p2_head = Point(3*self.w/4, self.h/2)
        self.p2_snake = [self.p2_head,
                        Point(self.p2_head.x+BLOCK_SIZE, self.p2_head.y),
                        Point(self.p2_head.x+(2*BLOCK_SIZE), self.p2_head.y)]
        self.p2_score = 0
        self.p2_alive = True
        
        # Food
        self.food = self.place_food([self.p1_snake, self.p2_snake])
        
        # Game state
        self.game_over = False
        self.winner = None
    
    def is_collision_pvp(self, point, snake1, snake2=None):
        # Hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        
        # Hits itself
        if point in snake1[1:]:
            return True
        
        # Hits other snake
        if snake2 and point in snake2:
                def is_collision_pvp(self, point, snake1, snake2):
        """Check for collisions in PvP mode"""
        # Hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # Hits own snake body (except head)
        if point in snake1[1:]:
            return True
        # Hits other snake (entire body including head)
        if point in snake2:
            return True
        return False
    
    def play_step(self):
        """Process one frame of the PvP game"""
        # 1. Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                # Player 1 Controls (WASD)
                if event.key == pygame.K_a and self.p1_direction != Direction.RIGHT:
                    self.p1_direction = Direction.LEFT
                elif event.key == pygame.K_d and self.p1_direction != Direction.LEFT:
                    self.p1_direction = Direction.RIGHT
                elif event.key == pygame.K_w and self.p1_direction != Direction.DOWN:
                    self.p1_direction = Direction.UP
                elif event.key == pygame.K_s and self.p1_direction != Direction.UP:
                    self.p1_direction = Direction.DOWN
                
                # Player 2 Controls (Arrow Keys)
                elif event.key == pygame.K_LEFT and self.p2_direction != Direction.RIGHT:
                    self.p2_direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.p2_direction != Direction.LEFT:
                    self.p2_direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.p2_direction != Direction.DOWN:
                    self.p2_direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.p2_direction != Direction.UP:
                    self.p2_direction = Direction.DOWN
                
                # Escape to quit
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        # 2. Move players if they're alive
        if self.p1_alive:
            # Move player 1
            new_head, self.p1_direction = self.move_snake(self.p1_direction, self.p1_head)
            self.p1_head = new_head
            self.p1_snake.insert(0, self.p1_head)
            
            # Check P1 collision
            if self.is_collision_pvp(self.p1_head, self.p1_snake, self.p2_snake):
                self.p1_alive = False
                if 'game_over' in self.sounds:
                    self.sounds['game_over'].play()
            
            # P1 eats food
            if self.p1_head == self.food:
                self.p1_score += 1
                self.food = self.place_food([self.p1_snake, self.p2_snake])
                if 'eat' in self.sounds:
                    self.sounds['eat'].play()
            else:
                self.p1_snake.pop()
        
        if self.p2_alive:
            # Move player 2
            new_head, self.p2_direction = self.move_snake(self.p2_direction, self.p2_head)
            self.p2_head = new_head
            self.p2_snake.insert(0, self.p2_head)
            
            # Check P2 collision
            if self.is_collision_pvp(self.p2_head, self.p2_snake, self.p1_snake):
                self.p2_alive = False
                if 'game_over' in self.sounds:
                    self.sounds['game_over'].play()
            
            # P2 eats food
            if self.p2_head == self.food:
                self.p2_score += 1
                self.food = self.place_food([self.p1_snake, self.p2_snake])
                if 'eat' in self.sounds:
                    self.sounds['eat'].play()
            else:
                self.p2_snake.pop()
        
        # Check game over condition
        if not self.p1_alive and not self.p2_alive:
            # Both died at the same time
            if self.p1_score > self.p2_score:
                self.winner = "Player 1"
            elif self.p2_score > self.p1_score:
                self.winner = "Player 2"
            else:
                self.winner = "Draw"
            self.game_over = True
            if 'win' in self.sounds:
                self.sounds['win'].play()
        elif not self.p1_alive:
            self.winner = "Player 2"
            self.game_over = True
            if 'win' in self.sounds:
                self.sounds['win'].play()
        elif not self.p2_alive:
            self.winner = "Player 1"
            self.game_over = True
            if 'win' in self.sounds:
                self.sounds['win'].play()
        
        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED_PVP)
    
    def _update_ui(self):
        """Render the PvP game display"""
        self.display.fill(BLACK)
        
        # Draw snakes
        if self.p1_alive:
            for pt in self.p1_snake:
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        if self.p2_alive:
            for pt in self.p2_snake:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Draw food
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw scores
        p1_text = self.font.render(f"P1(WASD): {self.p1_score}", True, GREEN1)
        p2_text = self.font.render(f"P2(Arrows): {self.p2_score}", True, BLUE1)
        self.display.blit(p1_text, [0, 0])
        self.display.blit(p2_text, [self.w - p2_text.get_width(), 0])
        
        pygame.display.flip()
    
    def show_game_over(self):
        """Show game over screen with winner information"""
        self.display.fill(BLACK)
        font = pygame.font.SysFont('arial', 40)
        
        if self.winner == "Draw":
            title_text = font.render("It's a Draw!", True, WHITE)
        else:
            title_text = font.render(f"{self.winner} Wins!", True, WHITE)
            
        score_text = font.render(f"P1: {self.p1_score} - P2: {self.p2_score}", True, WHITE)
        instruction_text = font.render("Press SPACE to play again or ESC to exit", True, WHITE)
        
        self.display.blit(title_text, [self.w/2 - title_text.get_width()/2, self.h/2 - 80])
        self.display.blit(score_text, [self.w/2 - score_text.get_width()/2, self.h/2 - 20])
        self.display.blit(instruction_text, [self.w/2 - instruction_text.get_width()/2, self.h/2 + 40])
        
        pygame.display.flip()
    
def main():
    game = SnakePvP()
    
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
