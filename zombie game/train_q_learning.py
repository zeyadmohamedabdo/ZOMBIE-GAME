import numpy as np
from zombie_env_short import ZombieEnvironment
from q_learning_agent import QLearningAgent
import matplotlib.pyplot as plt

def train(episodes=5000):
    # Create environment and agent
    env = ZombieEnvironment()
    agent = QLearningAgent(
        state_size=(env.grid_size, env.grid_size, 6),
        action_size=env.action_space.n,
        learning_rate=0.2,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Training statistics
    rewards_history = []
    steps_history = []
    best_reward = float('-inf')
    max_steps_per_episode = 200  # Maximum steps per episode
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps_per_episode:
            # Choose and perform action
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)
            
            # Learn from the action
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render every 100 episodes for visualization
            if episode % 100 == 0:
                env.render()
        
        # Record statistics
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        # Update best reward
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_q_table()  # Save the best Q-table
        
        # Print progress every 5 episodes
        if episode % 5 == 0:
            print(f"Episode: {episode}/{episodes}")
            print(f"Total Reward: {total_reward}")
            print(f"Steps: {steps}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Best Reward: {best_reward}")
            print("--------------------")
        
        # If we've achieved a good result, we can stop early
        if total_reward > 5000:  # Successfully completed the game
            print("Successfully solved the environment!")
            break
    
    env.close()
    return rewards_history, steps_history

def plot_results(rewards, steps):
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot steps
    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title('Episode Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rewards, steps = train()
    plot_results(rewards, steps) 