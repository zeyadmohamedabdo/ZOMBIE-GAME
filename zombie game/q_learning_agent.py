import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.2, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table as a dictionary
        self.q_table = {}
    
    def _get_state_key(self, state):
        # Find player position
        player_pos = None
        zombie_positions = []
        exit_pos = None
        wall_positions = []
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j, 0] == 1:  # Player
                    player_pos = (i, j)
                for k in range(1, 4):  # Zombies
                    if state[i, j, k] == 1:
                        zombie_positions.append((i, j, k-1))  # k-1 is the zombie index
                if state[i, j, 4] == 1:  # Exit
                    exit_pos = (i, j)
                if state[i, j, 5] == 1:  # Walls
                    wall_positions.append((i, j))
        
        # Calculate relative positions to player
        relative_positions = []
        
        # Add relative positions of zombies (sorted by level)
        for z_pos in sorted(zombie_positions, key=lambda x: x[2]):
            dx = z_pos[0] - player_pos[0]
            dy = z_pos[1] - player_pos[1]
            relative_positions.append((dx, dy))
        
        # Add relative position of exit if revealed
        if exit_pos is not None:
            dx = exit_pos[0] - player_pos[0]
            dy = exit_pos[1] - player_pos[1]
            relative_positions.append((dx, dy))
        
        # Add relative positions of nearby walls (within 2 cells)
        nearby_walls = []
        for wall_pos in wall_positions:
            dx = wall_pos[0] - player_pos[0]
            dy = wall_pos[1] - player_pos[1]
            if abs(dx) <= 2 and abs(dy) <= 2:  # Only consider walls within 2 cells
                nearby_walls.append((dx, dy))
        relative_positions.extend(sorted(nearby_walls))  # Add sorted wall positions
        
        return str(relative_positions)
    
    def choose_action(self, state):
        state_key = self._get_state_key(state)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # If state not in Q-table, initialize it
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize Q-values if states not in Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        if done:
            next_q = reward
        else:
            next_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])
        
        # Update Q-value
        self.q_table[state_key][action] = current_q + self.learning_rate * (next_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_q_table(self, filename='q_table.npy'):
        np.save(filename, dict(self.q_table))
    
    def load_q_table(self, filename='q_table.npy'):
        try:
            self.q_table = np.load(filename, allow_pickle=True).item()
            print("Loaded Q-table from", filename)
        except:
            print("No saved Q-table found, starting fresh") 