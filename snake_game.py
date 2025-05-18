#!/usr/bin/env python3

import sys
import os

# Make src directory importable for this script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snake_utils import Direction, Point, load_sounds

def main_menu():
    """Display main menu and launch selected game mode"""
    import pygame
    
    # Initialize pygame
    pygame.init()
    pygame.mixer.init()
    
    # Set up display
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Snake Game Collection')
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    
    # Load font
    font_large = pygame.font.SysFont('arial', 50)
    font_medium = pygame.font.SysFont('arial', 30)
    
    # Create menu items
    title = font_large.render('Snake Game Collection', True, WHITE)
    option1 = font_medium.render('1. Classic Snake (Single Player)', True, WHITE)
    option2 = font_medium.render('2. Snake PvP (WASD vs Arrows)', True, WHITE)
    option3 = font_medium.render('3. Human vs AI', True, WHITE)
    exit_option = font_medium.render('4. Exit', True, WHITE)
    
    # Instructions
    instruction = font_medium.render('Press number key to select', True, WHITE)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    pygame.quit()
                    from src.snake_classic import SnakeClassic
                    game = SnakeClassic()
                    game.run()
                    return
                
                elif event.key == pygame.K_2:
                    pygame.quit()
                    from src.snake_pvp import SnakePvP
                    game = SnakePvP()
                    game.run()
                    return
                
                elif event.key == pygame.K_3:
                    pygame.quit()
                    from src.snake_human_vs_ai import SnakeHumanVsAI
                    game = SnakeHumanVsAI()
                    game.run()
                    return
                
                elif event.key == pygame.K_4:
                    pygame.quit()
                    sys.exit()
        
        # Fill screen
        screen.fill(BLACK)
        
        # Draw menu items
        screen.blit(title, (width/2 - title.get_width()/2, 50))
        screen.blit(option1, (width/2 - option1.get_width()/2, 150))
        screen.blit(option2, (width/2 - option2.get_width()/2, 200))
        screen.blit(option3, (width/2 - option3.get_width()/2, 250))
        screen.blit(exit_option, (width/2 - exit_option.get_width()/2, 300))
        screen.blit(instruction, (width/2 - instruction.get_width()/2, 400))
        
        # Update display
        pygame.display.flip()

if __name__ == "__main__":
    main_menu()
