import pygame
import torch
import random
import numpy as np
import sys
import os
from src.core import SnakeGame, Direction, Point, BLOCK_SIZE
from src.core import BLACK, WHITE, GREEN1, GREEN2, BLUE1, BLUE2, RED, SPEED_AI
from src.core import draw_snake, draw_food, is_collision
from src.model import Linear_QNet

class SnakeHumanVsAI(SnakeGame):
    def __init__(self, w=640, h=480):
        super().__init__(w, h, "Snake - Human vs AI")
        
        # Initialize AI model
        self.model = Linear_QNet(11, 256, 3)
        model_path = os.path.join('models', 'model.pth')
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                print("AI model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("No AI model found, AI will make random moves")
        
        self.reset()
    
    def reset(self):
        """Reset the game state for both human and AI players"""
        # Human player (green)
        self.human_direction = Direction.RIGHT
        self.human_head = Point(self.w/4, self.h/2)
        self.human_snake = [self.human_head,
                          Point(self.human_head.x-BLOCK_SIZE, self.human_head.y),
                          Point(self.human_head.x-(2*BLOCK_SIZE), self.human_head.y)]
        self.human_score = 0
        self.human_alive = True
        
        # AI player (blue)
        self.ai_direction = Direction.LEFT
        self.ai_head = Point(3*self.w/4, self.h/2)
        self.ai_snake = [self.ai_head,
                        Point(self.ai_head.x+BLOCK_SIZE, self.ai_head.y),
                        Point(self.ai_head.x+(2*BLOCK_SIZE), self.ai_head.y)]
        self.ai_score = 0
        self.ai_alive = True
        
        # Food
        self.food = self.place_food([self.human_snake, self.ai_snake])
        
        # Game state
        self.game_over = False
        self.winner = None
        self.frame_iteration = 0
    
    def is_collision_ai(self, point):
        """Check for AI snake collisions"""
        # Hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        
        # Hits itself
        if point in self.ai_snake[1:]:
            return True
        
        # Hits human snake
        if point in self.human_snake:
            return True
            
        return False
        
    def is_collision_human(self, point):
        """Check for human snake collisions"""
        # Hits boundary or own body
        if is_collision(point, self.human_snake, self.w, self.h):
            return True
        
        # Hits AI snake
        if point in self.ai_snake:
            return True
            
        return False
    
    def get_state(self):
        """Get current state for AI decision making"""
        head = self.ai_snake[0]
        
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = self.ai_direction == Direction.LEFT
        dir_r = self.ai_direction == Direction.RIGHT
        dir_u = self.ai_direction == Direction.UP
        dir_d = self.ai_direction == Direction.DOWN

        # Create state vector
        state = [
            # Danger straight
            (dir_r and self.is_collision_ai(point_r)) or 
            (dir_l and self.is_collision_ai(point_l)) or 
            (dir_u and self.is_collision_ai(point_u)) or 
            (dir_d and self.is_collision_ai(point_d)),

            # Danger right
            (dir_u and self.is_collision_ai(point_r)) or 
            (dir_d and self.is_collision_ai(point_l)) or 
            (dir_l and self.is_collision_ai(point_u)) or 
            (dir_r and self.is_collision_ai(point_d)),

            # Danger left
            (dir_d and self.is_collision_ai(point_r)) or 
            (dir_u and self.is_collision_ai(point_l)) or 
            (dir_r and self.is_collision_ai(point_u)) or 
            (dir_l and self.is_collision_ai(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y   # food down
        ]
        
        return np.array(state, dtype=int)
    
    def get_ai_action(self):
        # Get current state
        state = self.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float)
        
        # Predict action
        with torch.no_grad():
            prediction = self.model(state_tensor)
        
        # Get move from prediction
        move = torch.argmax(prediction).item()
        
        # [straight, right, left]
        final_move = [0, 0, 0]
        final_move[move] = 1
        
        return final_move
    
    def move_ai(self, action):
        # [straight, right, left]
        turn_right = np.array_equal(action, [0, 1, 0])
        turn_left = np.array_equal(action, [0, 0, 1])
        
        new_head, new_dir = self.move_snake(
            self.ai_direction, 
            self.ai_head,
            turn_left,
            turn_right
        )
        
        self.ai_direction = new_dir
        self.ai_head = new_head
        if self.ai_direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.ai_direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.ai_direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.ai_direction == Direction.UP:
            y -= BLOCK_SIZE

        self.ai_head = Point(x, y)
    
    def move_human(self):
        new_head, self.human_direction = self.move_snake(
            self.human_direction, 
            self.human_head
        )
        self.human_head = new_head
    
    def play_step(self):
        self.frame_iteration += 1
        
        # 1. Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                # Human Controls (Arrow Keys)
                if event.key == pygame.K_LEFT and self.human_direction != Direction.RIGHT:
                    self.human_direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.human_direction != Direction.LEFT:
                    self.human_direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.human_direction != Direction.DOWN:
                    self.human_direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.human_direction != Direction.UP:
                    self.human_direction = Direction.DOWN
        
        # 2. Move players if they're alive
        # Move human player
        if self.human_alive:
            self.move_human()
            self.human_snake.insert(0, self.human_head)
            
            # Check collision
            if self.is_collision_human(self.human_head):
                self.human_alive = False
                if 'game_over' in self.sounds:
                    self.sounds['game_over'].play()
            
            # Human eats food
            if self.human_head == self.food:
                self.human_score += 1
                self.food = self.place_food([self.human_snake, self.ai_snake])
                if 'eat' in self.sounds:
                    self.sounds['eat'].play()
            else:
                self.human_snake.pop()
        
        # Move AI player
        if self.ai_alive:
            # Get AI action
            ai_action = self.get_ai_action()
            self.move_ai(ai_action)
            self.ai_snake.insert(0, self.ai_head)
            
            # Check timeout condition (AI is stuck in a loop)
            if self.frame_iteration > 100 * len(self.ai_snake):
                self.ai_alive = False
                if 'game_over' in self.sounds:
                    self.sounds['game_over'].play()
            
            # Check collision
            elif self.is_collision_ai(self.ai_head):
                self.ai_alive = False
                if 'game_over' in self.sounds:
                    self.sounds['game_over'].play()
            
            # AI eats food
            if self.ai_head == self.food:
                self.ai_score += 1
                self.food = self.place_food([self.human_snake, self.ai_snake])
                if 'eat' in self.sounds:
                    self.sounds['eat'].play()
            else:
                self.ai_snake.pop()
        
        # Check game over condition
        if not self.human_alive and not self.ai_alive:
            # Both died at the same time
            if self.human_score > self.ai_score:
                self.winner = "Human"
            elif self.ai_score > self.human_score:
                self.winner = "AI"
            else:
                self.winner = "Draw"
            self.game_over = True
            if 'win' in self.sounds:
                self.sounds['win'].play()
        elif not self.human_alive:
            self.winner = "AI"
            self.game_over = True
            if 'win' in self.sounds:
                self.sounds['win'].play()
        elif not self.ai_alive:
            self.winner = "Human"
            self.game_over = True
            if 'win' in self.sounds:
                self.sounds['win'].play()
        
        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED_AI)
    
    def _update_ui(self):
        """Render the game display"""
        self.display.fill(BLACK)
        
        # Draw snakes
        if self.human_alive:
            draw_snake(self.display, self.human_snake, GREEN1, GREEN2)
        
        if self.ai_alive:
            draw_snake(self.display, self.ai_snake, BLUE1, BLUE2)
        
        # Draw food
        draw_food(self.display, self.food)
        
        # Draw scores
        human_text = self.font.render(f"Human: {self.human_score}", True, GREEN1)
        ai_text = self.font.render(f"AI: {self.ai_score}", True, BLUE1)
        self.display.blit(human_text, [0, 0])
        self.display.blit(ai_text, [self.w - ai_text.get_width(), 0])
        
        pygame.display.flip()
    
    def show_game_over(self):
        """Show game over screen with winner information"""
        self.display.fill(BLACK)
        font = pygame.font.SysFont('arial', 40)
        
        if self.winner == "Draw":
            title_text = font.render("It's a Draw!", True, WHITE)
        else:
            title_text = font.render(f"{self.winner} Wins!", True, WHITE)
            
        score_text = font.render(f"Human: {self.human_score} - AI: {self.ai_score}", True, WHITE)
        instruction_text = font.render("Press SPACE to play again or ESC to exit", True, WHITE)
        
        self.display.blit(title_text, [self.w/2 - title_text.get_width()/2, self.h/2 - 80])
        self.display.blit(score_text, [self.w/2 - score_text.get_width()/2, self.h/2 - 20])
        self.display.blit(instruction_text, [self.w/2 - instruction_text.get_width()/2, self.h/2 + 40])
        
        pygame.display.flip()
    
def main():
    game = SnakeHumanVsAI()
    
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
