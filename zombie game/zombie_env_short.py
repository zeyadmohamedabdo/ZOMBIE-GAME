import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import time
import os

class ZombieEnvironment(gym.Env):
    def __init__(self, grid_size=8):
        super(ZombieEnvironment, self).__init__()
        
        self.grid_size = grid_size
        self.window_size = 800
        self.cell_size = (self.window_size - 200) // self.grid_size
        
        # Action space: 0: up, 1: right, 2: down, 3: left, 4: attack
        self.action_space = spaces.Discrete(5)
        
        # Observation space: grid_size x grid_size x 6 (player, zombie1, zombie10, zombie100, exit, walls)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.grid_size, self.grid_size, 6),
            dtype=np.float32
        )
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Castle Warrior RL")
        
        # Initialize fonts
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        
        # Movement delay (in seconds)
        self.delay = 0.001  # Much faster movement
        
        # Colors
        self.COLORS = {
            'background': (200, 200, 200),  # Stone gray
            'grid': (150, 150, 150),        # Darker stone
            'player': (255, 215, 0),        # Gold
            'zombie1': (100, 100, 100),     # Stone zombie
            'zombie10': (150, 0, 0),        # Blood zombie
            'zombie100': (200, 0, 0),       # Demon zombie
            'exit': (139, 69, 19),          # Brown castle door
            'wall': (50, 50, 50),           # Dark gray for walls
            'text': (0, 0, 0),
            'sidebar': (220, 220, 220)      # Light stone
        }
        
        # Fixed positions for zombies and exit
        self.fixed_zombie_positions = [
            (2, 2),  # Level 1 zombie
            (5, 5),  # Level 10 zombie
            (1, 6)   # Level 100 zombie
        ]
        self.fixed_exit_pos = (6, 1)
        
        # Load images
        self.load_images()
        
        self.reset()
    
    def load_images(self):
        # Create assets directory if it doesn't exist
        if not os.path.exists('assets'):
            os.makedirs('assets')
        
        # Try to load images, if they don't exist, create placeholders
        try:
            self.warrior_img = pygame.image.load('assets/warrior.png')
            self.warrior_img = pygame.transform.scale(self.warrior_img, (self.cell_size - 4, self.cell_size - 4))
        except:
            # Create a placeholder warrior icon
            self.warrior_img = pygame.Surface((self.cell_size - 4, self.cell_size - 4))
            self.warrior_img.fill(self.COLORS['player'])
            pygame.draw.rect(self.warrior_img, (0, 0, 0), (0, 0, self.cell_size - 4, self.cell_size - 4), 2)
        
        try:
            self.zombie1_img = pygame.image.load('assets/zombie1.png')
            self.zombie1_img = pygame.transform.scale(self.zombie1_img, (self.cell_size - 4, self.cell_size - 4))
        except:
            self.zombie1_img = pygame.Surface((self.cell_size - 4, self.cell_size - 4))
            self.zombie1_img.fill(self.COLORS['zombie1'])
        
        try:
            self.zombie10_img = pygame.image.load('assets/zombie10.png')
            self.zombie10_img = pygame.transform.scale(self.zombie10_img, (self.cell_size - 4, self.cell_size - 4))
        except:
            self.zombie10_img = pygame.Surface((self.cell_size - 4, self.cell_size - 4))
            self.zombie10_img.fill(self.COLORS['zombie10'])
        
        try:
            self.zombie100_img = pygame.image.load('assets/zombie100.png')
            self.zombie100_img = pygame.transform.scale(self.zombie100_img, (self.cell_size - 4, self.cell_size - 4))
        except:
            self.zombie100_img = pygame.Surface((self.cell_size - 4, self.cell_size - 4))
            self.zombie100_img.fill(self.COLORS['zombie100'])
        
        try:
            self.exit_img = pygame.image.load('assets/castle_door.png')
            self.exit_img = pygame.transform.scale(self.exit_img, (self.cell_size - 4, self.cell_size - 4))
        except:
            self.exit_img = pygame.Surface((self.cell_size - 4, self.cell_size - 4))
            self.exit_img.fill(self.COLORS['exit'])
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Initialize state with an extra channel for walls
        self.state = np.zeros((self.grid_size, self.grid_size, 6))
        
        # Create a maze-like structure with walls
        # Fixed positions that should not have walls
        protected_positions = [
            (0, 0),  # Player
            (0, self.grid_size-1),  # L1 zombie
            (self.grid_size-1, self.grid_size-1),  # L10 zombie
            (self.grid_size-1, 0),  # L100 zombie
            (self.grid_size//2, self.grid_size//2)  # Exit
        ]
        
        # Create walls in a maze-like pattern
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Don't place walls on protected positions
                if (i, j) not in protected_positions:
                    # Create walls in a pattern that allows paths but creates challenges
                    if (i % 2 == 0 and j % 3 == 0) or (i % 3 == 0 and j % 2 == 0):
                        self.state[i, j, 5] = 1  # Place wall
        
        # Fixed positions
        self.player_pos = (0, 0)  # Upper left
        self.state[0, 0, 0] = 1  # Player
        
        # Place zombies at fixed positions
        self.zombie_positions = []
        self.zombie_levels = [1, 10, 100]
        self.alive_zombies = [True, True, True]  # Track which zombies are still alive
        
        # Level 1 zombie in upper right
        self.zombie_positions.append((0, self.grid_size-1))
        self.state[0, self.grid_size-1, 1] = 1
        
        # Level 10 zombie in lower right
        self.zombie_positions.append((self.grid_size-1, self.grid_size-1))
        self.state[self.grid_size-1, self.grid_size-1, 2] = 1
        
        # Level 100 zombie in lower left
        self.zombie_positions.append((self.grid_size-1, 0))
        self.state[self.grid_size-1, 0, 3] = 1
        
        # Set fixed exit position in middle
        self.exit_pos = (self.grid_size//2, self.grid_size//2)
        
        self.exit_revealed = False
        self.steps = 0
        self.total_reward = 0
        return self.state, {}
    
    def _get_random_position(self):
        return (
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        )
    
    def _is_position_occupied(self, pos):
        # Check all channels including walls
        return np.any(self.state[pos[0], pos[1]] == 1)
    
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def step(self, action):
        self.steps += 1
        reward = -0.5  # Smaller negative reward for each step
        done = False
        info = {"action": action}
        
        # Store old position
        old_pos = self.player_pos
        new_pos = list(old_pos)
        
        # Move player
        if action < 4:  # Movement actions
            if action == 0:  # up
                new_pos[0] = max(0, old_pos[0] - 1)
            elif action == 1:  # right
                new_pos[1] = min(self.grid_size - 1, old_pos[1] + 1)
            elif action == 2:  # down
                new_pos[0] = min(self.grid_size - 1, old_pos[0] + 1)
            elif action == 3:  # left
                new_pos[1] = max(0, old_pos[1] - 1)
            
            # Check if new position is valid (not occupied by zombie or wall)
            can_move = True
            # Check for walls
            if self.state[new_pos[0], new_pos[1], 5] == 1:
                can_move = False
                reward -= 1  # Penalty for hitting wall
            
            # Check for zombies
            if can_move:
                for i, (pos, alive) in enumerate(zip(self.zombie_positions, self.alive_zombies)):
                    if tuple(new_pos) == pos and alive:
                        can_move = False
                        break
            
            if can_move:
                # Update player position
                self.state[old_pos[0], old_pos[1], 0] = 0
                self.state[new_pos[0], new_pos[1], 0] = 1
                self.player_pos = tuple(new_pos)
                
                # Give larger reward for moving towards correct zombie
                nearest_dist = float('inf')
                target_zombie_idx = 0
                if not self.alive_zombies[0]:
                    target_zombie_idx = 1
                if not self.alive_zombies[0] and not self.alive_zombies[1]:
                    target_zombie_idx = 2
                
                if self.alive_zombies[target_zombie_idx]:
                    dist = self._manhattan_distance(new_pos, self.zombie_positions[target_zombie_idx])
                    if dist < self._manhattan_distance(old_pos, self.zombie_positions[target_zombie_idx]):
                        reward += 5  # Bigger reward for moving towards target
                
                # If all zombies dead, reward moving towards exit
                if not any(self.alive_zombies) and self.exit_revealed:
                    if self._manhattan_distance(new_pos, self.exit_pos) < self._manhattan_distance(old_pos, self.exit_pos):
                        reward += 10
        
        elif action == 4:  # Attack action
            # Check if there's a zombie adjacent to the player
            for i, zombie_pos in enumerate(self.zombie_positions):
                if (self._manhattan_distance(self.player_pos, zombie_pos) == 1 and 
                    self.alive_zombies[i]):
                    # Check if we can kill this zombie (correct order)
                    if i == 0 or (i == 1 and not self.alive_zombies[0]) or (i == 2 and not self.alive_zombies[0] and not self.alive_zombies[1]):
                        self.alive_zombies[i] = False
                        self.state[zombie_pos[0], zombie_pos[1], i + 1] = 0
                        reward = self.zombie_levels[i] * 20  # Even bigger rewards for killing
                        info["killed_zombie"] = i
                        
                        # Reveal exit if all zombies are dead
                        if not any(self.alive_zombies):
                            self.exit_revealed = True
                            self.state[self.exit_pos[0], self.exit_pos[1], 4] = 1
                            reward += 500  # Big reward for killing all zombies
                    else:
                        reward = -200  # Bigger penalty for wrong order
                        done = True
        
        # Check if player reached the exit
        if self.exit_revealed and tuple(self.player_pos) == self.exit_pos:
            reward += 5000  # Much bigger completion bonus
            done = True
        
        # End episode if too many steps
        if self.steps >= 100:
            done = True
        
        self.total_reward += reward
        self.render(info)
        time.sleep(1.5)  # Even slower for better visualization
        
        return self.state, reward, done, False, info
    
    def render(self, info=None):
        # Fill background with stone texture
        self.screen.fill(self.COLORS['background'])
        
        # Draw castle grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.screen, self.COLORS['grid'],
                           (0, i * self.cell_size),
                           (self.window_size - 200, i * self.cell_size), 3)
            pygame.draw.line(self.screen, self.COLORS['grid'],
                           (i * self.cell_size, 0),
                           (i * self.cell_size, self.window_size), 3)
        
        # Draw walls
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.state[i, j, 5] == 1:  # Wall
                    wall_rect = pygame.Rect(
                        j * self.cell_size,
                        i * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(self.screen, self.COLORS['wall'], wall_rect)
        
        # Draw sidebar with stone texture
        sidebar_rect = pygame.Rect(self.window_size - 200, 0, 200, self.window_size)
        pygame.draw.rect(self.screen, self.COLORS['sidebar'], sidebar_rect)
        
        # Draw exit if revealed
        if self.exit_revealed:
            exit_rect = pygame.Rect(
                self.exit_pos[1] * self.cell_size + 2,
                self.exit_pos[0] * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4
            )
            self.screen.blit(self.exit_img, exit_rect)
        
        # Draw warrior
        warrior_rect = pygame.Rect(
            self.player_pos[1] * self.cell_size + 2,
            self.player_pos[0] * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        self.screen.blit(self.warrior_img, warrior_rect)
        
        # Draw zombies with their respective icons
        zombie_images = [self.zombie1_img, self.zombie10_img, self.zombie100_img]
        for i, (pos, alive) in enumerate(zip(self.zombie_positions, self.alive_zombies)):
            if alive:
                zombie_rect = pygame.Rect(
                    pos[1] * self.cell_size + 2,
                    pos[0] * self.cell_size + 2,
                    self.cell_size - 4,
                    self.cell_size - 4
                )
                self.screen.blit(zombie_images[i], zombie_rect)
                
                # Draw level indicator
                level_text = self.font.render(f"L{self.zombie_levels[i]}", True, self.COLORS['text'])
                text_rect = level_text.get_rect(center=(
                    pos[1] * self.cell_size + self.cell_size // 2,
                    pos[0] * self.cell_size - 15
                ))
                self.screen.blit(level_text, text_rect)
        
        # Draw sidebar information
        y_offset = 20
        # Title
        title = self.title_font.render("Castle Status", True, self.COLORS['text'])
        self.screen.blit(title, (self.window_size - 190, y_offset))
        
        y_offset += 50
        # Steps
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLORS['text'])
        self.screen.blit(steps_text, (self.window_size - 190, y_offset))
        
        y_offset += 30
        # Total Reward
        reward_text = self.font.render(f"Gold: {self.total_reward}", True, self.COLORS['text'])
        self.screen.blit(reward_text, (self.window_size - 190, y_offset))
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit() 