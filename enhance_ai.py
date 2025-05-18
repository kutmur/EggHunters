import os
import torch
import pygame
import time
import subprocess
from model import Linear_QNet
import numpy as np

print("\nEggHunters Snake Game Suite - AI Enhancement Utility")
print("===================================================\n")

# Check if enhanced model exists, if not, train it
model_folder_path = './model'
enhanced_model_path = os.path.join(model_folder_path, 'model_enhanced.pth')
basic_model_path = os.path.join(model_folder_path, 'model.pth')

if not os.path.exists(enhanced_model_path):
    print("Enhanced AI model not found. You need to train it first.")
    print("Starting enhanced training (this will improve AI's ability to compete against humans)...")
    print("Note: You can stop training at any time by pressing Ctrl+C when AI is performing well enough.\n")
    
    try:
        # Run the enhanced training script
        subprocess.run(["python", "train_enhanced.py"], check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Using the latest saved model.")
    except Exception as e:
        print(f"\nError during training: {e}")
        if not os.path.exists(enhanced_model_path):
            print("No enhanced model was created. Please try again.")
            exit(1)
else:
    print(f"Enhanced AI model found at {enhanced_model_path}\n")

# Backup the original model if it exists
if os.path.exists(basic_model_path):
    backup_path = os.path.join(model_folder_path, 'model_basic_backup.pth')
    if not os.path.exists(backup_path):
        try:
            torch.save(torch.load(basic_model_path), backup_path)
            print(f"Original AI model backed up to {backup_path}")
        except Exception as e:
            print(f"Warning: Could not backup original model: {e}")

# Update the snake_suite.py to use the enhanced model
print("\nUpdating snake_suite.py to use the enhanced AI model...")

try:
    # First, check if the model structure needs updating
    enhanced_model = torch.load(enhanced_model_path)
    model_has_13_inputs = False
    
    if 'linear1.weight' in enhanced_model:
        input_size = enhanced_model['linear1.weight'].shape[1]
        if input_size == 13:
            model_has_13_inputs = True
            print(f"Enhanced model has 13 inputs (includes human player awareness)")
        else:
            print(f"Enhanced model has {input_size} inputs")
    
    # Modify the Agent class in snake_suite.py to use 13 inputs when needed
    with open('snake_suite.py', 'r') as file:
        snake_suite_content = file.read()
    
    if model_has_13_inputs and "def get_state_for_human(self, game, human_snake, ai_snake)" not in snake_suite_content:
        print("Updating Agent class in snake_suite.py to handle enhanced model...")
        
        # Replace the Agent class initialization
        if "self.model = Linear_QNet(11, 256, 3)" in snake_suite_content:
            snake_suite_content = snake_suite_content.replace(
                "self.model = Linear_QNet(11, 256, 3)", 
                "self.model = Linear_QNet(13, 256, 3)  # Enhanced model with human awareness"
            )
        
        # Add the get_state_for_human method after get_state method
        get_state_code = '''
    def get_state_for_human(self, game, human_snake, ai_snake):
        """Genişletilmiş durum alma - insan yılanının pozisyonunu içerir"""
        head = ai_snake.snake[0]
        
        # Kontrol noktaları
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Yılanın yönü
        dir_l = ai_snake.direction == Direction.LEFT
        dir_r = ai_snake.direction == Direction.RIGHT
        dir_u = ai_snake.direction == Direction.UP
        dir_d = ai_snake.direction == Direction.DOWN
        
        # Çarpışma kontrolü için yardımcı fonksiyon
        def is_collision(pt):
            # Duvarlar
            if pt.x >= game.width or pt.x < 0 or pt.y >= game.height or pt.y < 0:
                return True
            # Kendi vücudu
            if pt in ai_snake.snake[1:]:
                return True
            # İnsan yılanı
            if pt in human_snake.snake:
                return True
            return False
        
        state = [
            # Tehlike düz
            (dir_r and is_collision(point_r)) or 
            (dir_l and is_collision(point_l)) or 
            (dir_u and is_collision(point_u)) or 
            (dir_d and is_collision(point_d)),
            
            # Tehlike sağ
            (dir_u and is_collision(point_r)) or 
            (dir_d and is_collision(point_l)) or 
            (dir_l and is_collision(point_u)) or 
            (dir_r and is_collision(point_d)),
            
            # Tehlike sol
            (dir_d and is_collision(point_r)) or 
            (dir_u and is_collision(point_l)) or 
            (dir_r and is_collision(point_u)) or 
            (dir_l and is_collision(point_d)),
            
            # Hareket yönü
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Yemek konumu
            game.food.x < head.x,  # yemek sol
            game.food.x > head.x,  # yemek sağ
            game.food.y < head.y,  # yemek yukarı
            game.food.y > head.y,  # yemek aşağı
        ]
        
        # İnsan yılanının pozisyonunu da duruma ekle
        human_head = human_snake.snake[0]
        
        # İnsan yılanının başı AI\'ın önünde mi?
        human_in_front = (
            (dir_r and human_head.x > head.x) or
            (dir_l and human_head.x < head.x) or
            (dir_u and human_head.y < head.y) or
            (dir_d and human_head.y > head.y)
        )
        
        # İnsan yılanı yakında mı?
        human_is_close = abs(head.x - human_head.x) + abs(head.y - human_head.y) < 5 * BLOCK_SIZE
        
        # Durumu güncelle
        state.extend([human_in_front, human_is_close])
        
        return np.array(state, dtype=int)'''
            
        # Insert after the get_state method
        if "def get_state(self, game, snake):" in snake_suite_content:
            get_state_index = snake_suite_content.find("def get_state(self, game, snake):")
            get_state_end = snake_suite_content.find("\n\n", get_state_index)
            if get_state_end > 0:
                snake_suite_content = snake_suite_content[:get_state_end] + get_state_code + snake_suite_content[get_state_end:]
        
        # Update the human vs AI game mode to use the enhanced state function
        if "state_old = agent.get_state(game, ai_snake)" in snake_suite_content:
            snake_suite_content = snake_suite_content.replace(
                "state_old = agent.get_state(game, ai_snake)",
                "state_old = agent.get_state_for_human(game, human_snake, ai_snake)"
            )
            
            # May need to update the other state_new line too
            if "state_new = agent.get_state(game, ai_snake)" in snake_suite_content:
                snake_suite_content = snake_suite_content.replace(
                    "state_new = agent.get_state(game, ai_snake)",
                    "state_new = agent.get_state_for_human(game, human_snake, ai_snake)"
                )
        
        # Update the model loading to use enhanced model
        model_loading_code = '''
    # Eğitilmiş modeli yükleme
    enhanced_model_path = './model/model_enhanced.pth'
    basic_model_path = './model/model.pth'
    
    if os.path.exists(enhanced_model_path):
        agent.model.load_state_dict(torch.load(enhanced_model_path))
        agent.model.eval()  # Değerlendirme moduna geçir
        print("AI için geliştirilmiş model yüklendi!")
    elif os.path.exists(basic_model_path):
        agent.model.load_state_dict(torch.load(basic_model_path))
        agent.model.eval()
        print("AI için basit model yüklendi.")
    else:
        print("Eğitilmiş model bulunamadı, rastgele hareketler kullanılacak.")'''
            
        if "# Eğitilmiş modeli yükleme\n    model_path = './model/model.pth'" in snake_suite_content:
            snake_suite_content = snake_suite_content.replace(
                "# Eğitilmiş modeli yükleme\n    model_path = './model/model.pth'\n    if os.path.exists(model_path):\n        agent.model.load_state_dict(torch.load(model_path))\n        agent.model.eval()  # Değerlendirme moduna geçir\n        print(\"AI için eğitilmiş model yüklendi!\")\n    else:\n        print(\"Eğitilmiş model bulunamadı, rastgele hareketler kullanılacak.\")",
                model_loading_code
            )
        
        # Add numpy import if needed
        if "import numpy as np" not in snake_suite_content:
            import_line = snake_suite_content.find("import")
            import_end = snake_suite_content.find("\n\n", import_line)
            if import_end > 0:
                snake_suite_content = snake_suite_content[:import_end] + "\nimport numpy as np" + snake_suite_content[import_end:]
        
        # Write the updated content back to the file
        with open('snake_suite.py', 'w') as file:
            file.write(snake_suite_content)
        
        print("snake_suite.py successfully updated to use the enhanced AI model!")
    else:
        if model_has_13_inputs:
            print("snake_suite.py already configured for enhanced model")
        else:
            print("Enhanced model doesn't have the expected 13 inputs, keeping original format")
    
    print("\nSuccess! Your AI Snake has been enhanced!")
    print("Start the game with 'python snake_suite.py' and try the 'Human vs AI' mode.")
    print("The AI should now be better at competing against human players.")

except Exception as e:
    print(f"\nError updating snake_suite.py: {e}")
    print("You can still manually copy model_enhanced.pth to model.pth to use the enhanced AI.")

# Update the snake_suite.py to use the enhanced model
print("\nUpdating snake_suite.py to use the enhanced AI model...")

try:
    # First, check if the model structure needs updating
    enhanced_model = torch.load(enhanced_model_path)
    model_has_13_inputs = False
    
    if 'linear1.weight' in enhanced_model:
        input_size = enhanced_model['linear1.weight'].shape[1]
        if input_size == 13:
            model_has_13_inputs = True
            print(f"Enhanced model has 13 inputs (includes human player awareness)")
        else:
            print(f"Enhanced model has {input_size} inputs")        # Modify the Agent class in snake_suite.py to use 13 inputs when needed
    with open('snake_suite.py', 'r') as file:
        snake_suite_content = file.read()
    
    if model_has_13_inputs and "def get_state_for_human(self, game, human_snake, ai_snake)" not in snake_suite_content:
        print("Updating Agent class in snake_suite.py to handle enhanced model...")
        
        # Replace the Agent class initialization
        if "self.model = Linear_QNet(11, 256, 3)" in snake_suite_content:
            snake_suite_content = snake_suite_content.replace(
                "self.model = Linear_QNet(11, 256, 3)", 
                "self.model = Linear_QNet(13, 256, 3)  # Enhanced model with human awareness"
            )
        
        # Add the get_state_for_human method after get_state method
        if "def get_state(self, game, snake):" in snake_suite_content:
            get_state_code = """
    def get_state_for_human(self, game, human_snake, ai_snake):
        \"\"\"Genişletilmiş durum alma - insan yılanının pozisyonunu içerir\"\"\"
        head = ai_snake.body[0]
        
        # Kontrol noktaları
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Yılanın yönü
        dir_l = ai_snake.direction == Point(-1, 0)
        dir_r = ai_snake.direction == Point(1, 0)
        dir_u = ai_snake.direction == Point(0, -1)
        dir_d = ai_snake.direction == Point(0, 1)
        
        # Çarpışma kontrolü için yardımcı fonksiyon
        def is_collision(point):
            # Duvarlar
            if point.x >= game.width or point.x < 0 or point.y >= game.height or point.y < 0:
                return True
            # Kendi vücudu
            if point in ai_snake.body[1:]:
                return True
            # İnsan yılanı
            if point in human_snake.body:
                return True
            return False
        
        state = [
            # Tehlike düz
            (dir_r and is_collision(point_r)) or 
            (dir_l and is_collision(point_l)) or 
            (dir_u and is_collision(point_u)) or 
            (dir_d and is_collision(point_d)),
            
            # Tehlike sağ
            (dir_u and is_collision(point_r)) or 
            (dir_d and is_collision(point_l)) or 
            (dir_l and is_collision(point_u)) or 
            (dir_r and is_collision(point_d)),
            
            # Tehlike sol
            (dir_d and is_collision(point_r)) or 
            (dir_u and is_collision(point_l)) or
            (dir_r and is_collision(point_u)) or 
            (dir_l and is_collision(point_d)),
            
            # Hareket yönü
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Yemek konumu
            game.food.x < head.x,  # yemek sol
            game.food.x > head.x,  # yemek sağ
            game.food.y < head.y,  # yemek yukarı
            game.food.y > head.y,  # yemek aşağı
        ]
        
        # İnsan yılanının pozisyonunu da duruma ekle
        human_head = human_snake.body[0]
        
        # İnsan yılanının başı AI'ın önünde mi?
        human_in_front = (
            (dir_r and human_head.x > head.x) or
            (dir_l and human_head.x < head.x) or
            (dir_u and human_head.y < head.y) or
            (dir_d and human_head.y > head.y)
        )
        
        # İnsan yılanı yakında mı?
        human_is_close = abs(head.x - human_head.x) + abs(head.y - human_head.y) < 5 * BLOCK_SIZE
        
        # Durumu güncelle
        state.extend([human_in_front, human_is_close])
        
        return np.array(state, dtype=int)"""
            
            # Insert after the get_state method
            get_state_index = snake_suite_content.find("def get_state(self, game, snake):")
            get_state_end = snake_suite_content.find("\n\n", get_state_index)
            if get_state_end > 0:
                snake_suite_content = snake_suite_content[:get_state_end] + get_state_code + snake_suite_content[get_state_end:]
        
        # Update the human vs AI game mode to use the enhanced state function
        if "state_old = agent.get_state(game, ai_snake)" in snake_suite_content:
            snake_suite_content = snake_suite_content.replace(
                "state_old = agent.get_state(game, ai_snake)",
                "state_old = agent.get_state_for_human(game, human_snake, ai_snake)"
            )
        
        # Update the model loading to use enhanced model
        if "model_path = './model/model.pth'" in snake_suite_content:
            model_loading_code = """
    # Eğitilmiş modeli yükleme
    enhanced_model_path = './model/model_enhanced.pth'
    basic_model_path = './model/model.pth'
    
    if os.path.exists(enhanced_model_path):
        agent.model.load_state_dict(torch.load(enhanced_model_path))
        agent.model.eval()  # Değerlendirme moduna geçir
        print("AI için geliştirilmiş model yüklendi!")
    elif os.path.exists(basic_model_path):
        agent.model.load_state_dict(torch.load(basic_model_path))
        agent.model.eval()
        print("AI için basit model yüklendi.")
    else:
        print("Eğitilmiş model bulunamadı, rastgele hareketler kullanılacak.")"""
            
            snake_suite_content = snake_suite_content.replace(
                "# Eğitilmiş modeli yükleme\n    model_path = './model/model.pth'\n    if os.path.exists(model_path):\n        agent.model.load_state_dict(torch.load(model_path))\n        agent.model.eval()  # Değerlendirme moduna geçir\n        print(\"AI için eğitilmiş model yüklendi!\")\n    else:\n        print(\"Eğitilmiş model bulunamadı, rastgele hareketler kullanılacak.\")",
                model_loading_code
            )
        
        # Write the updated content back to the file
        with open('snake_suite.py', 'w') as file:
            file.write(snake_suite_content)
        
        print("snake_suite.py successfully updated to use the enhanced AI model!")
    else:
        if model_has_13_inputs:
            print("snake_suite.py already configured for enhanced model")
        else:
            print("Enhanced model doesn't have the expected 13 inputs, keeping original format")
    
    print("\nSuccess! Your AI Snake has been enhanced!")
    print("Start the game with 'python snake_suite.py' and try the 'Human vs AI' mode.")
    print("The AI should now be better at competing against human players.")

except Exception as e:
    print(f"\nError updating snake_suite.py: {e}")
    print("You can still manually copy model_enhanced.pth to model.pth to use the enhanced AI.")
