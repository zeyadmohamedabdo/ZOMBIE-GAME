import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import time

class ZombieEnvironment(gym.Env):
    def __init__(self, grid_size=8):
        super(ZombieEnvironment, self).__init__()
        
        self.grid_size = grid_size
        self.window_size = 800  # Increased window size
        self.cell_size = (self.window_size - 200) // self.grid_size  # Adjust for sidebar
        
        # Action space: 0: up, 1: right, 2: down, 3: left, 4: attack
        self.action_space = spaces.Discrete(5)
        
        # Observation space: grid_size x grid_size x 5 (player, zombie1, zombie10, zombie100, exit)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.grid_size, self.grid_size, 5),
            dtype=np.float32
        )
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Zombie Soldier RL")
        
        # Initialize fonts
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        
        # Movement delay (in seconds)
        self.delay = 1.0  # Increased delay for better visualization
        
        # Colors
        self.COLORS = {
            'background': (255, 255, 255),
            'grid': (200, 200, 200),
            'player': (0, 0, 255),
            'zombie1': (100, 100, 100),
            'zombie10': (150, 0, 0),
            'zombie100': (200, 0, 0),
            'exit': (0, 255, 0),
            'text': (0, 0, 0),
            'sidebar': (240, 240, 240)
        }
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Initialize state
        self.state = np.zeros((self.grid_size, self.grid_size, 5))
        
        # Place player randomly
        self.player_pos = self._get_random_position()
        self.state[self.player_pos[0], self.player_pos[1], 0] = 1
        
        # Place zombies randomly (ensuring they're not on the player or each other)
        self.zombie_positions = []
        self.zombie_levels = [1, 10, 100]
        self.alive_zombies = [True, True, True]  # Track which zombies are still alive
        
        for i in range(3):
            while True:
                pos = self._get_random_position()
                if not self._is_position_occupied(pos):
                    self.zombie_positions.append(pos)
                    self.state[pos[0], pos[1], i + 1] = 1
                    break
        
        # Initialize exit position (will be revealed later)
        while True:
            self.exit_pos = self._get_random_position()
            if not self._is_position_occupied(self.exit_pos):
                break
        
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
            
            # Check if new position is valid (not occupied by zombie)
            can_move = True
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
        # Fill background
        self.screen.fill(self.COLORS['background'])
        
        # Draw sidebar
        sidebar_rect = pygame.Rect(self.window_size - 200, 0, 200, self.window_size)
        pygame.draw.rect(self.screen, self.COLORS['sidebar'], sidebar_rect)
        
        # Draw game area
        game_area = pygame.Rect(0, 0, self.window_size - 200, self.window_size)
        pygame.draw.rect(self.screen, self.COLORS['background'], game_area)
        
        # Draw grid lines
        for i in range(self.grid_size):
            pygame.draw.line(self.screen, self.COLORS['grid'],
                           (0, i * self.cell_size),
                           (self.window_size - 200, i * self.cell_size))
            pygame.draw.line(self.screen, self.COLORS['grid'],
                           (i * self.cell_size, 0),
                           (i * self.cell_size, self.window_size))
        
        # Draw exit if revealed
        if self.exit_revealed:
            exit_center = (self.exit_pos[1] * self.cell_size + self.cell_size // 2,
                         self.exit_pos[0] * self.cell_size + self.cell_size // 2)
            pygame.draw.rect(self.screen, self.COLORS['exit'],
                           (self.exit_pos[1] * self.cell_size + 5,
                            self.exit_pos[0] * self.cell_size + 5,
                            self.cell_size - 10, self.cell_size - 10))
            exit_text = self.font.render("EXIT", True, self.COLORS['text'])
            text_rect = exit_text.get_rect(center=exit_center)
            self.screen.blit(exit_text, text_rect)
        
        # Draw player
        pygame.draw.circle(self.screen, self.COLORS['player'],
                         (self.player_pos[1] * self.cell_size + self.cell_size // 2,
                          self.player_pos[0] * self.cell_size + self.cell_size // 2),
                         self.cell_size // 3)
        
        # Draw zombies with level indicators
        zombie_colors = [self.COLORS['zombie1'], self.COLORS['zombie10'], self.COLORS['zombie100']]
        for i, (pos, alive) in enumerate(zip(self.zombie_positions, self.alive_zombies)):
            if alive:
                # Draw zombie
                pygame.draw.rect(self.screen, zombie_colors[i],
                               (pos[1] * self.cell_size + self.cell_size // 4,
                                pos[0] * self.cell_size + self.cell_size // 4,
                                self.cell_size // 2, self.cell_size // 2))
                
                # Draw level indicator
                level_text = self.font.render(f"L{self.zombie_levels[i]}", True, self.COLORS['text'])
                text_rect = level_text.get_rect(center=(pos[1] * self.cell_size + self.cell_size // 2,
                                                      pos[0] * self.cell_size + self.cell_size // 2))
                self.screen.blit(level_text, text_rect)
        
        # Draw sidebar information
        y_offset = 20
        # Title
        title = self.title_font.render("Game Status", True, self.COLORS['text'])
        self.screen.blit(title, (self.window_size - 190, y_offset))
        
        y_offset += 50
        # Steps
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLORS['text'])
        self.screen.blit(steps_text, (self.window_size - 190, y_offset))
        
        y_offset += 30
        # Total Reward
        reward_text = self.font.render(f"Total Reward: {self.total_reward:.1f}", True, self.COLORS['text'])
        self.screen.blit(reward_text, (self.window_size - 190, y_offset))
        
        y_offset += 50
        # Zombie Status
        status_title = self.font.render("Zombies Remaining:", True, self.COLORS['text'])
        self.screen.blit(status_title, (self.window_size - 190, y_offset))
        
        for i, alive in enumerate(self.alive_zombies):
            y_offset += 25
            status = "Alive" if alive else "Defeated"
            color = self.COLORS['text'] if alive else (100, 100, 100)
            zombie_text = self.font.render(f"Level {self.zombie_levels[i]}: {status}", True, color)
            self.screen.blit(zombie_text, (self.window_size - 190, y_offset))
        
        # Last action
        if info and "action" in info:
            y_offset += 50
            action_names = ["Up", "Right", "Down", "Left", "Attack"]
            action_text = self.font.render(f"Action: {action_names[info['action']]}", True, self.COLORS['text'])
            self.screen.blit(action_text, (self.window_size - 190, y_offset))
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit() 