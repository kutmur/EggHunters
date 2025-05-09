import pygame
import random
import sys
import time
import math
from pygame import mixer
from pygame import gfxdraw

# Initialize Pygame and mixer
pygame.init()
mixer.init()

# Constants
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE
EGG_LIFETIME = 5  # seconds
EXPLOSION_RADIUS = 5
BOMB_LIFETIME = 5  # seconds
SCORCHED_EARTH_DURATION = 5  # seconds
TELEPORT_GATE_DURATION = 10  # seconds
HATCHING_DURATION = 3  # seconds
SHELL_PROTECTION_EGGS = 3  # Number of eggs needed to lose shell

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
BROWN = (139, 69, 19)  # Brown for scorched earth
GRASS_GREEN = (144, 238, 144)  # Fresh grass green
BACKGROUND = GRASS_GREEN
GRID_COLOR = (134, 228, 134)  # Slightly darker than background
TELEPORT_COLOR = (100, 200, 255)
SHELL_COLOR = (200, 200, 200)

# Animation constants
SNAKE_GROWTH_DURATION = 0.3
SNAKE_FLASH_DURATION = 0.2
EXPLOSION_DURATION = 0.5
EGG_CRACK_STAGES = 5
TELEPORT_DURATION = 0.5
SHELL_DROP_DURATION = 1.0

class TeleportGate:
    def __init__(self, position):
        self.position = position
        self.creation_time = time.time()
        self.pulse_phase = 0
        self.active = True

    def update(self):
        self.pulse_phase = (time.time() * 2) % (2 * math.pi)
        self.active = time.time() - self.creation_time < TELEPORT_GATE_DURATION

    def draw(self, screen):
        if not self.active:
            return

        x, y = self.position
        center_x = x * GRID_SIZE + GRID_SIZE // 2
        center_y = y * GRID_SIZE + GRID_SIZE // 2
        size = GRID_SIZE - 4

        # Draw pulsing rings
        for i in range(3):
            ring_size = size * (1 + 0.2 * math.sin(self.pulse_phase + i * math.pi/3))
            alpha = int(150 * (1 - i/3))
            color = (*TELEPORT_COLOR, alpha)
            surf = pygame.Surface((ring_size * 2, ring_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (ring_size, ring_size), ring_size, width=2)
            screen.blit(surf, (center_x - ring_size, center_y - ring_size))

        # Draw center portal
        pygame.draw.circle(screen, TELEPORT_COLOR, (center_x, center_y), size//2)
        
        # Draw swirling particles
        particle_count = 8
        for i in range(particle_count):
            angle = 2 * math.pi * i / particle_count + time.time() * 3
            distance = size * 0.3 * (1 + 0.2 * math.sin(self.pulse_phase))
            particle_x = center_x + math.cos(angle) * distance
            particle_y = center_y + math.sin(angle) * distance
            particle_size = 2
            pygame.draw.circle(screen, WHITE, (particle_x, particle_y), particle_size)

class Explosion:
    def __init__(self, position):
        self.position = position
        self.radius = EXPLOSION_RADIUS
        self.active = True
        self.creation_time = time.time()
        self.duration = 2  # seconds
        self.scorched_earth = True  # Enable scorched earth effect
        self.scorched_tiles = self.get_danger_zone()

    def is_active(self):
        return time.time() - self.creation_time < self.duration

    def get_scorched_fade_factor(self):
        # Returns 1.0 at creation time, fades to 0.0 over SCORCHED_EARTH_DURATION
        elapsed = time.time() - self.creation_time
        if elapsed < self.duration:  # During active explosion, full strength
            return 1.0
        elif elapsed > SCORCHED_EARTH_DURATION:  # After fade duration, gone
            return 0.0
        else:  # During fade period, linear fade
            return 1.0 - ((elapsed - self.duration) / (SCORCHED_EARTH_DURATION - self.duration))

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
        self.growth_time = 0
        self.is_growing = False
        self.growth_progress = 0
        self.head_angle = 0  # For smooth head rotation
        self.hatching_progress = 0
        self.is_hatching = True
        self.hatch_start_time = time.time()
        self.has_shell = True
        self.eggs_eaten = 0
        self.shell_drop_progress = 0
        self.is_dropping_shell = False
        self.teleport_progress = 0
        self.is_teleporting = False
        self.teleport_target = None

    def move(self):
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.body.insert(0, new_head)
        self.body.pop()
        
        # Update head angle based on direction
        if self.direction == (1, 0):  # Right
            self.head_angle = 0
        elif self.direction == (-1, 0):  # Left
            self.head_angle = 180
        elif self.direction == (0, -1):  # Up
            self.head_angle = 90
        elif self.direction == (0, 1):  # Down
            self.head_angle = 270

    def grow(self):
        self.body.append(self.body[-1])
        self.score += 1
        self.is_flashing = True
        self.flash_time = time.time()
        self.is_growing = True
        self.growth_time = time.time()
        self.growth_progress = 0
        self.eggs_eaten += 1

        # Check if shell should drop
        if self.has_shell and self.eggs_eaten >= SHELL_PROTECTION_EGGS:
            self.has_shell = False
            self.is_dropping_shell = True
            self.shell_drop_time = time.time()
            self.shell_drop_progress = 0

    def teleport(self, target_pos):
        self.is_teleporting = True
        self.teleport_start_time = time.time()
        self.teleport_progress = 0
        self.teleport_target = target_pos

    def update_animations(self):
        current_time = time.time()
        
        # Update hatching animation
        if self.is_hatching:
            self.hatching_progress = min(1.0, 
                (current_time - self.hatch_start_time) / HATCHING_DURATION)
            if self.hatching_progress >= 1.0:
                self.is_hatching = False

        # Update growth animation
        if self.is_growing:
            self.growth_progress = min(1.0, 
                (current_time - self.growth_time) / SNAKE_GROWTH_DURATION)
            if self.growth_progress >= 1.0:
                self.is_growing = False

        # Update shell drop animation
        if self.is_dropping_shell:
            self.shell_drop_progress = min(1.0,
                (current_time - self.shell_drop_time) / SHELL_DROP_DURATION)
            if self.shell_drop_progress >= 1.0:
                self.is_dropping_shell = False

        # Update teleport animation
        if self.is_teleporting:
            self.teleport_progress = min(1.0,
                (current_time - self.teleport_start_time) / TELEPORT_DURATION)
            if self.teleport_progress >= 1.0:
                self.is_teleporting = False
                self.body[0] = self.teleport_target
                self.teleport_target = None

        # Update flash animation
        if self.is_flashing and current_time - self.flash_time >= SNAKE_FLASH_DURATION:
            self.is_flashing = False

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
        if self.is_flashing and time.time() - self.flash_time < SNAKE_FLASH_DURATION:
            return WHITE
        return self.color

    def draw(self, screen):
        # Draw body segments with smooth corners
        for i, segment in enumerate(self.body):
            x, y = segment
            center_x = x * GRID_SIZE + GRID_SIZE // 2
            center_y = y * GRID_SIZE + GRID_SIZE // 2
            
            # Calculate size based on growth animation
            size = GRID_SIZE - 2
            if i == len(self.body) - 1 and self.is_growing:
                size = int(size * (1 + 0.5 * math.sin(self.growth_progress * math.pi)))
            
            # Draw rounded rectangle for body segments
            rect = pygame.Rect(
                center_x - size//2,
                center_y - size//2,
                size,
                size
            )
            pygame.draw.rect(screen, self.get_color(), rect, border_radius=size//2)
            
            # Draw head with direction indicator
            if i == 0:
                # Draw shell if present
                if self.has_shell:
                    shell_size = int(size * 1.2)
                    shell_rect = pygame.Rect(
                        center_x - shell_size//2,
                        center_y - shell_size//2,
                        shell_size,
                        shell_size
                    )
                    pygame.draw.rect(screen, SHELL_COLOR, shell_rect, border_radius=shell_size//2)
                
                # Draw eyes
                eye_offset = size // 4
                eye_size = size // 6
                eye_color = BLACK
                
                # Calculate eye positions based on direction
                if self.direction == (1, 0):  # Right
                    left_eye = (center_x + eye_offset, center_y - eye_offset)
                    right_eye = (center_x + eye_offset, center_y + eye_offset)
                elif self.direction == (-1, 0):  # Left
                    left_eye = (center_x - eye_offset, center_y - eye_offset)
                    right_eye = (center_x - eye_offset, center_y + eye_offset)
                elif self.direction == (0, -1):  # Up
                    left_eye = (center_x - eye_offset, center_y - eye_offset)
                    right_eye = (center_x + eye_offset, center_y - eye_offset)
                else:  # Down
                    left_eye = (center_x - eye_offset, center_y + eye_offset)
                    right_eye = (center_x + eye_offset, center_y + eye_offset)
                
                pygame.draw.circle(screen, eye_color, left_eye, eye_size)
                pygame.draw.circle(screen, eye_color, right_eye, eye_size)

        # Draw teleport effect
        if self.is_teleporting:
            head = self.body[0]
            center_x = head[0] * GRID_SIZE + GRID_SIZE // 2
            center_y = head[1] * GRID_SIZE + GRID_SIZE // 2
            
            # Draw swirling particles
            particle_count = 12
            for i in range(particle_count):
                angle = 2 * math.pi * i / particle_count + time.time() * 5
                distance = GRID_SIZE * (0.5 + self.teleport_progress)
                particle_x = center_x + math.cos(angle) * distance
                particle_y = center_y + math.sin(angle) * distance
                particle_size = 3
                alpha = int(255 * (1 - self.teleport_progress))
                color = (*TELEPORT_COLOR, alpha)
                pygame.draw.circle(screen, color, (particle_x, particle_y), particle_size)

        # Draw shell drop animation
        if self.is_dropping_shell:
            head = self.body[0]
            center_x = head[0] * GRID_SIZE + GRID_SIZE // 2
            center_y = head[1] * GRID_SIZE + GRID_SIZE // 2
            
            # Calculate shell position (falling down)
            shell_y = center_y + GRID_SIZE * self.shell_drop_progress
            shell_size = int(GRID_SIZE * (1 - 0.5 * self.shell_drop_progress))
            
            # Draw shell with crack effect
            shell_rect = pygame.Rect(
                center_x - shell_size//2,
                shell_y - shell_size//2,
                shell_size,
                shell_size
            )
            pygame.draw.rect(screen, SHELL_COLOR, shell_rect, border_radius=shell_size//2)
            
            # Draw cracks
            crack_color = (100, 100, 100)
            crack_width = 1
            for _ in range(3):
                start_x = center_x - shell_size//2 + random.randint(0, shell_size)
                start_y = shell_y - shell_size//2 + random.randint(0, shell_size)
                end_x = start_x + random.randint(-shell_size//2, shell_size//2)
                end_y = start_y + random.randint(-shell_size//2, shell_size//2)
                pygame.draw.line(screen, crack_color, (start_x, start_y), (end_x, end_y), crack_width)

class Egg:
    def __init__(self):
        self.position = self.generate_position()
        self.creation_time = time.time()
        self.is_exploding = False
        self.crack_stage = 0
        self.shake_offset = (0, 0)
        self.last_shake_time = time.time()

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
                self.crack_stage = 0
                self.shake_offset = (0, 0)
                break

    def should_explode(self):
        return time.time() - self.creation_time >= EGG_LIFETIME

    def update(self):
        if self.is_exploding:
            return

        current_time = time.time()
        time_alive = current_time - self.creation_time
        
        # Update crack stage based on time
        self.crack_stage = min(EGG_CRACK_STAGES - 1,
                             int(time_alive / (EGG_LIFETIME / EGG_CRACK_STAGES)))
        
        # Update shake animation in the last 30% of lifetime
        if time_alive > EGG_LIFETIME * 0.7:
            if current_time - self.last_shake_time > 0.1:  # Update every 0.1 seconds
                self.shake_offset = (
                    random.randint(-2, 2),
                    random.randint(-2, 2)
                )
                self.last_shake_time = current_time

    def draw(self, screen):
        if self.is_exploding:
            return

        x, y = self.position
        center_x = x * GRID_SIZE + GRID_SIZE // 2 + self.shake_offset[0]
        center_y = y * GRID_SIZE + GRID_SIZE // 2 + self.shake_offset[1]
        size = GRID_SIZE - 4

        # Draw egg base with light shading
        egg_color = WHITE  # White egg as requested
        egg_highlight = (255, 255, 255)  # Pure white for highlights
        egg_shadow = (220, 220, 220)     # Light gray for shadows
        
        # Draw main egg shape with gradient
        pygame.draw.ellipse(screen, egg_color,
                          (center_x - size//2, center_y - size//2,
                           size, size))
                           
        # Add subtle highlight (top-left)
        highlight_size = int(size * 0.6)
        highlight_offset = int(size * 0.2)
        pygame.draw.ellipse(screen, egg_highlight,
                         (center_x - size//2 + highlight_offset, 
                          center_y - size//2 + highlight_offset,
                          highlight_size, highlight_size), 1)

        # Draw cracks based on stage
        if self.crack_stage > 0:
            crack_color = (100, 100, 100)
            crack_width = 1
            
            # Draw random cracks that increase with each stage
            seed = hash(self.position) % 1000  # Use position to seed deterministic randomness
            random.seed(seed)
            
            for i in range(self.crack_stage * 3):  # More cracks per stage
                # Create a zigzag crack line
                points = []
                start_angle = random.uniform(0, math.pi * 2)
                start_dist = random.uniform(0.2, 0.8) * (size // 2)
                points.append((
                    center_x + math.cos(start_angle) * start_dist,
                    center_y + math.sin(start_angle) * start_dist
                ))
                
                # Create a few more points for the zigzag
                last_angle = start_angle
                for _ in range(random.randint(2, 4)):
                    angle = last_angle + random.uniform(-math.pi/4, math.pi/4)
                    dist = random.uniform(0.1, 0.3) * size
                    points.append((
                        points[-1][0] + math.cos(angle) * dist,
                        points[-1][1] + math.sin(angle) * dist
                    ))
                    last_angle = angle
                
                # Draw the crack line
                if len(points) > 1:
                    pygame.draw.lines(screen, crack_color, False, points, crack_width)
            
            random.seed()  # Reset random seed

class Bomb:
    def __init__(self):
        self.position = self.generate_position()
        self.creation_time = time.time()
        self.is_exploding = False
        self.explosion_progress = 0
        self.pulse_offset = 0  # For pulsing effect

    def generate_position(self):
        x = random.randint(2, GRID_WIDTH - 3)
        y = random.randint(2, GRID_HEIGHT - 3)
        return (x, y)

    def respawn(self, snake1, snake2, egg):
        while True:
            new_pos = self.generate_position()
            if (new_pos not in snake1.body and 
                new_pos not in snake2.body and
                new_pos != egg.position):
                self.position = new_pos
                self.creation_time = time.time()
                self.is_exploding = False
                self.explosion_progress = 0
                self.pulse_offset = 0
                break

    def should_explode(self):
        return time.time() - self.creation_time >= BOMB_LIFETIME

    def update(self):
        current_time = time.time()
        if self.is_exploding:
            self.explosion_progress = min(1.0, 
                (current_time - self.creation_time - BOMB_LIFETIME) / EXPLOSION_DURATION)
            return self.explosion_progress >= 1.0
        else:
            # Update pulse effect
            self.pulse_offset = 0.25 * math.sin(current_time * 5)
            return False

    def draw(self, screen):
        x, y = self.position
        center_x = x * GRID_SIZE + GRID_SIZE // 2
        center_y = y * GRID_SIZE + GRID_SIZE // 2
        
        if self.is_exploding:
            # Draw explosion with particle effects
            radius = int(GRID_SIZE * (1 + self.explosion_progress * 3))
            
            # Create explosion particles
            particle_count = 20
            for i in range(particle_count):
                angle = 2 * math.pi * i / particle_count
                distance = radius * (0.5 + 0.5 * self.explosion_progress)
                particle_x = center_x + math.cos(angle) * distance
                particle_y = center_y + math.sin(angle) * distance
                
                # Size decreases as explosion progresses
                particle_size = int(GRID_SIZE * (1 - self.explosion_progress) * 0.5)
                
                # Color fades from orange to red
                color = (
                    255,
                    int(165 * (1 - self.explosion_progress)),
                    0,
                    int(255 * (1 - self.explosion_progress))
                )
                
                # Draw particle
                surf = pygame.Surface((particle_size * 2, particle_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (particle_size, particle_size), particle_size)
                screen.blit(surf, (particle_x - particle_size, particle_y - particle_size))
                
            # Draw explosion wave
            wave_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            wave_color = (255, 100, 0, int(150 * (1 - self.explosion_progress)))
            pygame.draw.circle(wave_surf, wave_color, (radius, radius), radius, width=3)
            screen.blit(wave_surf, (center_x - radius, center_y - radius))
        else:
            # Draw bomb with increasing warning effect
            time_alive = time.time() - self.creation_time
            progress = min(1.0, time_alive / BOMB_LIFETIME)
            size = GRID_SIZE - 4
            
            # Base size + pulse effect that gets stronger as bomb ages
            pulse_strength = 0.1 + 0.2 * progress
            size_with_pulse = size * (1 + self.pulse_offset * pulse_strength)
            
            # Interpolate color from yellow to red
            color = (
                255,  # Red stays at 255
                int(YELLOW[1] * (1 - progress) + RED[1] * progress),  # Green fades
                0,    # Blue stays at 0
            )
            
            # Draw bomb body with gradient
            pygame.draw.circle(screen, color, (center_x, center_y), size_with_pulse // 2)
            
            # Draw highlight
            highlight_size = int(size_with_pulse * 0.3)
            highlight_offset = int(size_with_pulse * 0.1)
            pygame.draw.circle(screen, (*color, 150), 
                             (center_x - highlight_offset, center_y - highlight_offset),
                             highlight_size, width=1)
            
            # Draw fuse with sparks
            fuse_length = int(size * (0.5 + 0.3 * progress))  # Fuse gets shorter
            fuse_angle = math.pi/4  # Fixed angle
            fuse_end_x = center_x - math.cos(fuse_angle) * fuse_length
            fuse_end_y = center_y - math.sin(fuse_angle) * fuse_length
            
            # Draw the fuse (thicker as time passes)
            fuse_width = 1 + int(progress * 2)
            pygame.draw.line(screen, (100, 70, 40), 
                           (center_x, center_y),
                           (fuse_end_x, fuse_end_y), fuse_width)
            
            # Draw sparks at end of fuse (more as time passes)
            spark_count = 1 + int(progress * 5)
            for _ in range(spark_count):
                spark_angle = fuse_angle + random.uniform(-0.5, 0.5)
                spark_distance = random.uniform(2, 6) * (1 + progress)
                spark_x = fuse_end_x - math.cos(spark_angle) * spark_distance
                spark_y = fuse_end_y - math.sin(spark_angle) * spark_distance
                spark_size = 1 + random.uniform(0, 2) * (1 + progress)
                
                # Sparks are yellow to orange
                spark_color = (
                    255,
                    int(255 * (0.5 + random.uniform(0, 0.5))),
                    0
                )
                pygame.draw.circle(screen, spark_color, (spark_x, spark_y), spark_size)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("EggHunters: Snake Battle")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.title_font = pygame.font.Font(None, 72)
        
        # Load sounds
        try:
            self.eat_sound = mixer.Sound('eat.wav')
            self.explosion_sound = mixer.Sound('explosion.wav')
            self.teleport_sound = mixer.Sound('teleport.wav')
            self.hatch_sound = mixer.Sound('hatch.wav')
        except:
            print("Sound files not found. Game will run without sound.")
            self.eat_sound = None
            self.explosion_sound = None
            self.teleport_sound = None
            self.hatch_sound = None
        
        # Initialize game objects
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
        self.bomb = Bomb()
        self.explosions = []
        self.scorched_areas = []
        self.teleport_gates = []
        self.game_over = False
        self.winner = None
        self.last_update = time.time()
        self.update_interval = 0.15
        self.start_time = time.time()
        self.last_teleport_spawn = time.time()
        self.teleport_spawn_interval = 5  # seconds

    def spawn_teleport_gate(self):
        # Spawn near egg with 50% chance
        if random.random() < 0.5:
            x, y = self.egg.position
            # Try positions around the egg
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and
                    (new_x, new_y) not in self.snake1.body and
                    (new_x, new_y) not in self.snake2.body):
                    self.teleport_gates.append(TeleportGate((new_x, new_y)))
                    return

    def update(self):
        if self.game_over:
            return

        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        self.last_update = current_time

        # Update animations
        self.snake1.update_animations()
        self.snake2.update_animations()
        self.egg.update()
        self.bomb.update()

        # Update teleport gates
        for gate in self.teleport_gates:
            gate.update()
        self.teleport_gates = [gate for gate in self.teleport_gates if gate.active]

        # Spawn new teleport gates
        if current_time - self.last_teleport_spawn >= self.teleport_spawn_interval:
            self.spawn_teleport_gate()
            self.last_teleport_spawn = current_time

        # Move snakes if they're not hatching
        if not self.snake1.is_hatching:
            self.snake1.move()
        if not self.snake2.is_hatching:
            self.snake2.move()

        # Check egg lifetime
        if self.egg.should_explode() and not self.egg.is_exploding:
            self.egg.is_exploding = True
            self.egg.respawn(self.snake1, self.snake2)

        # Check bomb lifetime
        if self.bomb.should_explode() and not self.bomb.is_exploding:
            self.bomb.is_exploding = True
            explosion = Explosion(self.bomb.position)
            self.explosions.append(explosion)
            self.scorched_areas.append({
                'tiles': explosion.scorched_tiles,
                'creation_time': current_time,
                'position': self.bomb.position
            })
            if self.explosion_sound:
                self.explosion_sound.play()

        # Update bomb
        if self.bomb.update():
            self.bomb.respawn(self.snake1, self.snake2, self.egg)

        # Remove expired explosions
        self.explosions = [exp for exp in self.explosions if exp.is_active()]

        # Check teleport collisions
        for gate in self.teleport_gates:
            if gate.active:
                # Check snake1
                if self.snake1.body[0] == gate.position and not self.snake1.is_teleporting:
                    # Find valid teleport target
                    while True:
                        target_x = random.randint(2, GRID_WIDTH - 3)
                        target_y = random.randint(2, GRID_HEIGHT - 3)
                        target_pos = (target_x, target_y)
                        if (target_pos not in self.snake1.body and
                            target_pos not in self.snake2.body and
                            target_pos != self.egg.position and
                            target_pos != self.bomb.position):
                            self.snake1.teleport(target_pos)
                            if self.teleport_sound:
                                self.teleport_sound.play()
                            break

                # Check snake2
                if self.snake2.body[0] == gate.position and not self.snake2.is_teleporting:
                    while True:
                        target_x = random.randint(2, GRID_WIDTH - 3)
                        target_y = random.randint(2, GRID_HEIGHT - 3)
                        target_pos = (target_x, target_y)
                        if (target_pos not in self.snake1.body and
                            target_pos not in self.snake2.body and
                            target_pos != self.egg.position and
                            target_pos != self.bomb.position):
                            self.snake2.teleport(target_pos)
                            if self.teleport_sound:
                                self.teleport_sound.play()
                            break

        # Check collisions with walls, self, other snake, and active explosions
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
        # Draw background
        self.screen.fill(BACKGROUND)
        
        # Draw grid lines with subtle effect
        for x in range(0, WINDOW_WIDTH, GRID_SIZE):
            alpha = int(30 * (1 + math.sin(x / 100) * 0.2))
            pygame.draw.line(self.screen, (*GRID_COLOR, alpha),
                           (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
            alpha = int(30 * (1 + math.sin(y / 100) * 0.2))
            pygame.draw.line(self.screen, (*GRID_COLOR, alpha),
                           (0, y), (WINDOW_WIDTH, y))
                            
        # Draw scorched earth areas
        for area in self.scorched_areas:
            fade_factor = 1.0
            elapsed = time.time() - area['creation_time']
            if elapsed > SCORCHED_EARTH_DURATION:
                fade_factor = 0
            elif elapsed > 2:  # After explosion is done, start fading
                fade_factor = 1.0 - ((elapsed - 2) / (SCORCHED_EARTH_DURATION - 2))
                
            if fade_factor > 0:
                for pos in area['tiles']:
                    x, y = pos
                    rect = pygame.Rect(
                        x * GRID_SIZE,
                        y * GRID_SIZE,
                        GRID_SIZE,
                        GRID_SIZE
                    )
                    # Mix brown with background color based on fade factor
                    color = (
                        int(BROWN[0] * fade_factor + BACKGROUND[0] * (1-fade_factor)),
                        int(BROWN[1] * fade_factor + BACKGROUND[1] * (1-fade_factor)),
                        int(BROWN[2] * fade_factor + BACKGROUND[2] * (1-fade_factor))
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    
                    # Add some dirt texture
                    for _ in range(3):
                        dot_x = x * GRID_SIZE + random.randint(2, GRID_SIZE-3)
                        dot_y = y * GRID_SIZE + random.randint(2, GRID_SIZE-3)
                        dot_size = random.randint(1, 3)
                        dot_color = (
                            max(0, color[0] - 30),
                            max(0, color[1] - 30),
                            max(0, color[2] - 30),
                            int(255 * fade_factor)
                        )
                        pygame.draw.circle(self.screen, dot_color, (dot_x, dot_y), dot_size)

        # Draw game objects
        self.egg.draw(self.screen)
        self.bomb.draw(self.screen)
        
        # Draw teleport gates
        for gate in self.teleport_gates:
            gate.draw(self.screen)
        
        # Draw active explosions on top of scorched earth
        for explosion in self.explosions:
            if explosion.active:
                radius = EXPLOSION_RADIUS * GRID_SIZE
                center_x = explosion.position[0] * GRID_SIZE + GRID_SIZE // 2
                center_y = explosion.position[1] * GRID_SIZE + GRID_SIZE // 2
                
                # Draw explosion wave rings
                for r in range(int(radius), 0, -5):
                    alpha = int(150 * (r / radius))
                    color = (255, 100, 0, alpha)
                    surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, color, (r, r), r, width=3)
                    self.screen.blit(surf, (center_x - r, center_y - r))
                
                # Draw fire particles
                particle_count = 20
                for i in range(particle_count):
                    angle = 2 * math.pi * i / particle_count + time.time() % 1.0
                    distance = random.uniform(0.2, 0.8) * radius
                    particle_x = center_x + math.cos(angle) * distance
                    particle_y = center_y + math.sin(angle) * distance
                    
                    particle_size = random.randint(2, 5)
                    particle_color = (
                        255, 
                        random.randint(50, 150),
                        0,
                        random.randint(100, 200)
                    )
                    
                    pygame.draw.circle(self.screen, particle_color, 
                                    (particle_x, particle_y), particle_size)
        
        # Draw snakes on top of everything
        self.snake1.draw(self.screen)
        self.snake2.draw(self.screen)

        # Draw scores with modern UI
        game_time = int(time.time() - self.start_time)
        minutes = game_time // 60
        seconds = game_time % 60
        time_text = f"{minutes:02d}:{seconds:02d}"
        
        score1_text = self.font.render(f"P1: {self.snake1.score}", True, GREEN)
        score2_text = self.font.render(f"P2: {self.snake2.score}", True, BLUE)
        time_display = self.font.render(time_text, True, WHITE)
        
        # Draw score backgrounds with rounded corners
        pygame.draw.rect(self.screen, (*GREEN, 80),
                        (10, 10, score1_text.get_width() + 20, score1_text.get_height() + 10),
                        border_radius=10)
        pygame.draw.rect(self.screen, (*BLUE, 80),
                        (WINDOW_WIDTH - score2_text.get_width() - 30, 10,
                         score2_text.get_width() + 20, score2_text.get_height() + 10),
                        border_radius=10)
        pygame.draw.rect(self.screen, (40, 40, 40, 80),
                        (WINDOW_WIDTH//2 - time_display.get_width()//2 - 10, 10,
                         time_display.get_width() + 20, time_display.get_height() + 10),
                        border_radius=10)
        
        self.screen.blit(score1_text, (20, 15))
        self.screen.blit(score2_text, (WINDOW_WIDTH - score2_text.get_width() - 20, 15))
        self.screen.blit(time_display, (WINDOW_WIDTH//2 - time_display.get_width()//2, 15))

        # Draw game over screen with modern UI
        if self.game_over:
            # Semi-transparent overlay
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            # Game over text
            if self.winner == "Draw":
                text = "Game Over - Draw!"
            else:
                text = f"Game Over - {self.winner} Wins!"
            
            game_over_text = self.title_font.render(text, True, WHITE)
            text_rect = game_over_text.get_rect(center=(WINDOW_WIDTH/2, WINDOW_HEIGHT/2 - 40))
            
            # Draw text with glow effect
            for offset in range(5, 0, -1):
                glow_text = self.title_font.render(text, True, (255, 255, 255, 30))
                glow_rect = glow_text.get_rect(center=(WINDOW_WIDTH/2 + offset,
                                                     WINDOW_HEIGHT/2 - 40 + offset))
                self.screen.blit(glow_text, glow_rect)
            
            self.screen.blit(game_over_text, text_rect)
            
            # Draw restart message
            restart_text = self.font.render("Press SPACE to restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH/2, WINDOW_HEIGHT/2 + 40))
            self.screen.blit(restart_text, restart_rect)

        pygame.display.flip()

    def handle_input(self):
        for event in pygame.event.get():
            # Quit the game
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            # Handle key presses
            if event.type == pygame.KEYDOWN:
                # Restart the game if it's game over
                if self.game_over and event.key == pygame.K_SPACE:
                    self.__init__()
                    return
                
                # Snake 1 controls
                if not self.snake1.is_hatching:
                    for key, direction in self.snake1.controls.items():
                        if event.key == key and (
                            self.snake1.direction[0] != -direction[0] or 
                            self.snake1.direction[1] != -direction[1]
                        ):
                            self.snake1.direction = direction
                            break
                
                # Snake 2 controls
                if not self.snake2.is_hatching:
                    for key, direction in self.snake2.controls.items():
                        if event.key == key and (
                            self.snake2.direction[0] != -direction[0] or 
                            self.snake2.direction[1] != -direction[1]
                        ):
                            self.snake2.direction = direction
                            break

    def run(self):
        while True:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60)  # Cap at 60 FPS for smooth animations

if __name__ == "__main__":
    game = Game()
    game.run()