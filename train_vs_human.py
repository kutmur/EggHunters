import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
from enum import Enum
import pygame
import os

# Renk tanımları
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 15  # Eğitim sırasında biraz daha hızlı

class AdvancedSnakeGameAI:
    """İki yılanlı (AI ve sanal insan) eğitim ortamı"""
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Ekranı başlat
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI Training vs Virtual Human')
        self.clock = pygame.time.Clock()
        self.reset()
        self.font = pygame.font.SysFont('arial', 25)

    def reset(self):
        # AI yılanı
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        # Sanal insan yılanı (karşı tarafta)
        self.human_direction = Direction.LEFT
        self.human_head = Point(self.w/4, self.h/2)
        self.human_snake = [self.human_head,
                           Point(self.human_head.x+BLOCK_SIZE, self.human_head.y),
                           Point(self.human_head.x+(2*BLOCK_SIZE), self.human_head.y)]

        self.score = 0
        self.human_score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.human_snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. Kullanıcı girdisini al
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. AI yılanını hareket ettir
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. Sanal insan yılanını hareket ettir - basit bir stratejiye göre
        self._move_human()
        self.human_snake.insert(0, self.human_head)
        
        # 4. Oyun sonu kontrolü
        reward = 0
        game_over = False
        
        # AI kendine veya duvara çarptı mı?
        if self.is_collision(self.head, self.snake[1:]) or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -15  # Daha büyük ceza
            return reward, game_over, self.score
        
        # AI, sanal insanın vücuduna çarptı mı?
        if self.head in self.human_snake:
            game_over = True
            reward = -20  # İnsan yılanına çarpma için daha büyük ceza
            return reward, game_over, self.score
        
        # Sanal insan AI'ın vücuduna çarptı mı? (Bu AI için bir ödüldür)
        if self.human_head in self.snake:
            reward = 25  # İnsan çarparsa AI için daha büyük ödül
            self.human_snake = [self.human_head]  # İnsan kaybeder, yeniden başlar
            
        # Sanal insan kendine veya duvara çarptı mı?
        if self.is_collision(self.human_head, self.human_snake[1:]):
            # İnsan kaybetti, kendine çarptı
            self.human_snake = [self.human_head]
        
        # 5. Yemek kontrolü
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            
        if self.human_head == self.food:
            self.human_score += 1
            reward = -5  # AI için ceza, çünkü insan yemeği aldı
            self._place_food()
        else:
            self.human_snake.pop()
        
        # İnsan yakınsa ve AI ondan kaçabiliyorsa, küçük bir ödül ver
        if abs(self.head.x - self.human_head.x) + abs(self.head.y - self.human_head.y) < 3 * BLOCK_SIZE:
            reward += 0.5  # İnsan yakın ve hala hayattaysak, bu iyi bir durum
        
        # 6. UI güncelleme
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 7. Oyun durumunu döndür
        return reward, game_over, self.score

    def is_collision(self, pt, body_parts):
        # Duvar çarpışması
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Kendine çarpışma
        if pt in body_parts:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # AI yılanını çiz
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Sanal insan yılanını çiz
        for pt in self.human_snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Yiyeceği çiz
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Skorları çiz
        text1 = self.font.render(f"AI Score: {self.score}", True, BLUE1)
        text2 = self.font.render(f"Human Score: {self.human_score}", True, GREEN1)
        self.display.blit(text1, [0, 0])
        self.display.blit(text2, [self.w-140, 0])
        
        pygame.display.flip()

    def _move(self, action):
        # AI hareketi - [düz, sağ, sol]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # değişiklik yok
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # sağa dön
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # sola dön

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        
    def _move_human(self):
        """Sanal insanın hareketi için basit bir strateji."""
        # Hedef: Yemeğe yönelirken AI'dan kaçma
        
        # 1. Potansiyel tehlikeleri belirle
        dangers = []
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.human_direction)
        
        # Doğrudan AI ile çarpışma kontrolü
        distance_to_ai = abs(self.human_head.x - self.head.x) + abs(self.human_head.y - self.head.y)
        ai_is_close = distance_to_ai < 3 * BLOCK_SIZE
        
        potential_moves = []
        
        # Tüm olası hareketleri değerlendir
        for i, direction in enumerate(clock_wise):
            # Mevcut konumdan test hareket
            test_x = self.human_head.x
            test_y = self.human_head.y
            
            if direction == Direction.RIGHT:
                test_x += BLOCK_SIZE
            elif direction == Direction.LEFT:
                test_x -= BLOCK_SIZE
            elif direction == Direction.DOWN:
                test_y += BLOCK_SIZE
            elif direction == Direction.UP:
                test_y -= BLOCK_SIZE
            
            test_pt = Point(test_x, test_y)
            # Bu hareket güvenli mi?
            is_safe = not (test_pt in self.human_snake[1:] or 
                           test_pt in self.snake or
                           test_x > self.w - BLOCK_SIZE or test_x < 0 or 
                           test_y > self.h - BLOCK_SIZE or test_y < 0)
                           
            if is_safe:
                # Yemeğe olan uzaklık
                food_distance = abs(test_x - self.food.x) + abs(test_y - self.food.y)
                # AI'ya olan uzaklık
                ai_distance = abs(test_x - self.head.x) + abs(test_y - self.head.y)
                
                score = 0
                # Yemeğe yaklaşmak iyidir
                score -= food_distance * 0.5
                # AI'dan uzaklaşmak iyidir (AI yakınsa)
                if ai_is_close:
                    score += ai_distance * 2
                
                potential_moves.append((score, direction))
        
        # Ters yöne gitmeyi engelle
        opposite_idx = (idx + 2) % 4
        safe_potential_moves = [m for m in potential_moves if m[1] != clock_wise[opposite_idx]]
        
        if not safe_potential_moves and potential_moves:
            # Tüm güvenli hareketleri kullandık, en iyi skoru seç
            potential_moves.sort(reverse=True)
            self.human_direction = potential_moves[0][1]
        elif safe_potential_moves:
            # Güvenli hareketlerden en iyi skoru seç
            safe_potential_moves.sort(reverse=True)
            self.human_direction = safe_potential_moves[0][1]
        # Else: Mevcut yönde devam et
        
        # Son hareketi uygula
        x = self.human_head.x
        y = self.human_head.y
        if self.human_direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.human_direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.human_direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.human_direction == Direction.UP:
            y -= BLOCK_SIZE

        self.human_head = Point(x, y)

        
def train_vs_human():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = AdvancedSnakeGameAI()
    
    # Eğitilmiş modeli kontrol et ve yükle
    model_folder_path = './model'
    model_path = os.path.join(model_folder_path, 'model.pth')
    
    # Eğitilmiş model varsa yükle
    if os.path.exists(model_path):
        try:
            model_state = torch.load(model_path)
            agent.model.load_state_dict(model_state)
            print(f"Mevcut model yüklendi: {model_path}")
            print("Eğitime devam ediliyor...")
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")
    else:
        print("Eğitilmiş model bulunamadı. Sıfırdan eğitime başlanıyor.")

    while True:
        # Eski durumu al
        state_old = agent.get_state(game)

        # Hamleyi belirle
        final_move = agent.get_action(state_old)

        # Hamleyi yap ve yeni durumu al
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Kısa belleği eğit
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Hafıza
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Uzun belleği eğit ve grafiği çiz
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Oyun', agent.n_games, 'Skor', score, 'Rekor:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


class Agent:
    """AI Agent sınıfı - orijinal agent.py'den alındı"""
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # indirim oranı
        self.memory = deque(maxlen=100_000)  # popleft()
        self.model = Linear_QNet(13, 256, 3)  # 13 inputs for enhanced state (includes human snake position)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)


    def get_state(self, game):
        """Oyun durumunu alma - AdvancedSnakeGameAI için özel sürüm."""
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Tehlike düz - kendisi, duvarlar ve insan yılanı
            (dir_r and (game.is_collision(point_r, game.snake[1:]) or point_r in game.human_snake)) or 
            (dir_l and (game.is_collision(point_l, game.snake[1:]) or point_l in game.human_snake)) or 
            (dir_u and (game.is_collision(point_u, game.snake[1:]) or point_u in game.human_snake)) or 
            (dir_d and (game.is_collision(point_d, game.snake[1:]) or point_d in game.human_snake)),

            # Tehlike sağ - kendisi, duvarlar ve insan yılanı
            (dir_u and (game.is_collision(point_r, game.snake[1:]) or point_r in game.human_snake)) or 
            (dir_d and (game.is_collision(point_l, game.snake[1:]) or point_l in game.human_snake)) or 
            (dir_l and (game.is_collision(point_u, game.snake[1:]) or point_u in game.human_snake)) or 
            (dir_r and (game.is_collision(point_d, game.snake[1:]) or point_d in game.human_snake)),

            # Tehlike sol - kendisi, duvarlar ve insan yılanı
            (dir_d and (game.is_collision(point_r, game.snake[1:]) or point_r in game.human_snake)) or 
            (dir_u and (game.is_collision(point_l, game.snake[1:]) or point_l in game.human_snake)) or 
            (dir_r and (game.is_collision(point_u, game.snake[1:]) or point_u in game.human_snake)) or 
            (dir_l and (game.is_collision(point_d, game.snake[1:]) or point_d in game.human_snake)),
            
            # Hareket yönü
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Yemek konumu
            game.food.x < head.x,  # yemek sol
            game.food.x > head.x,  # yemek sağ
            game.food.y < head.y,  # yemek yukarı
            game.food.y > head.y  # yemek aşağı
            ]

        # İnsan yılanının pozisyonunu da duruma ekle
        # İnsan yılanının başı AI'ın önünde mi?
        human_head = game.human_snake[0]
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
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random hareketler: keşif / sömürü dengesi
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


if __name__ == '__main__':
    train_vs_human()
