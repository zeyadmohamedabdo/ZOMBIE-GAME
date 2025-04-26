import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Initialize Q-table with a simple state representation
        # We'll use the relative positions of zombies to the player as state
        self.q_table = {}
    
    def _get_state_key(self, state):
        # Convert the state matrix to a more compact representation
        player_pos = None
        zombie_positions = []
        
        # Find player and zombie positions
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j, 0] == 1:  # Player
                    player_pos = (i, j)
                for k in range(1, 4):  # Zombies
                    if state[i, j, k] == 1:
                        zombie_positions.append((i, j, k-1))  # k-1 is the zombie index
        
        # Calculate relative positions
        relative_positions = []
        for z_pos in sorted(zombie_positions, key=lambda x: x[2]):  # Sort by zombie index
            if z_pos[2] == 0:  # Only include relative positions of alive zombies
                relative_positions.append((z_pos[0] - player_pos[0], z_pos[1] - player_pos[1]))
        
        return str(relative_positions)
    
    def choose_action(self, state):
        state_key = self._get_state_key(state)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
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
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        
        # Q-learning formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q * (1 - done) - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 