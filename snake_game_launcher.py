import sys
import pygame
import os
import tkinter as tk
from tkinter import ttk, messagebox

# Check if required directories exist, create them if not
for dir_path in ['assets', 'models']:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class SnakeGameLauncher:
    """Main launcher for all snake game variants"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Snake Game Launcher")
        self.root.geometry("500x400")
        self.root.configure(bg="#2c3e50")
        self.root.resizable(False, False)
        
        # Set up header
        header_frame = ttk.Frame(root)
        header_frame.pack(pady=20)
        
        title_label = ttk.Label(header_frame, text="Snake Game Launcher", 
                               font=("Helvetica", 24, "bold"))
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, text="Select a game mode to play", 
                                  font=("Helvetica", 14))
        subtitle_label.pack(pady=10)
        
        # Game mode selection frame
        mode_frame = ttk.Frame(root)
        mode_frame.pack(pady=20, fill="x")
        
        # Classic mode button
        self.create_mode_button(
            mode_frame, 
            "Classic Mode", 
            "Single player snake game", 
            self.launch_classic_mode
        )
        
        # PVP mode button
        self.create_mode_button(
            mode_frame, 
            "PvP Mode", 
            "2-Player: WASD vs Arrows", 
            self.launch_pvp_mode
        )
        
        # Human vs AI mode button
        self.create_mode_button(
            mode_frame, 
            "Human vs AI", 
            "Challenge the trained AI", 
            self.launch_human_vs_ai_mode
        )
        
        # Footer with exit button
        footer_frame = ttk.Frame(root)
        footer_frame.pack(pady=20)
        
        exit_button = ttk.Button(
            footer_frame, 
            text="Exit", 
            command=self.root.destroy
        )
        exit_button.pack(pady=10)
        
    def create_mode_button(self, parent, title, description, command):
        """Create a styled button for a game mode"""
        frame = ttk.Frame(parent)
        frame.pack(pady=10, padx=20, fill="x")
        
        button = ttk.Button(
            frame,
            text=title,
            command=command,
            width=20
        )
        button.pack(side="left", padx=10)
        
        desc_label = ttk.Label(
            frame,
            text=description,
            font=("Helvetica", 10)
        )
        desc_label.pack(side="left", padx=10)
        
    def launch_classic_mode(self):
        """Launch the classic snake game"""
        self.root.destroy()
        os.system(f"{sys.executable} -m src.snake_classic")
        
    def launch_pvp_mode(self):
        """Launch the PvP snake game"""
        self.root.destroy()
        os.system(f"{sys.executable} -m src.snake_pvp")
        
    def launch_human_vs_ai_mode(self):
        """Launch the Human vs AI snake game"""
        self.root.destroy()
        os.system(f"{sys.executable} -m src.snake_human_vs_ai")

if __name__ == "__main__":
    if not os.path.exists(os.path.join('models', 'model.pth')):
        print("Warning: AI model file not found. Human vs AI mode may use random movements.")
    
    root = tk.Tk()
    # Set style
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12))
    
    app = SnakeGameLauncher(root)
    root.mainloop()