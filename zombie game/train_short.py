from zombie_env import ZombieEnvironment
from q_learning_agent import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt

def train(episodes=2):  # Only 2 episodes
    # Create environment and agent
    env = ZombieEnvironment()
    agent = QLearningAgent(
        state_size=(env.grid_size, env.grid_size, 5),
        action_size=env.action_space.n,
        learning_rate=0.2,  # Higher learning rate
        discount_factor=0.99,  # High discount factor
        epsilon=0.3  # Start with low exploration (more exploitation)
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    best_reward = float('-inf')
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Choose and perform action
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)
            
            # Learn from the action
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Print progress every episode
        print(f"\nEpisode {episode + 1}/{episodes}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Steps: {steps}")
        print(f"Best Reward: {best_reward:.2f}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        print("--------------------")
        
        # If we've achieved a good result, we can stop early
        if episode_reward > 5000:  # Successfully completed the game
            print("Successfully solved the environment!")
            break
    
    env.close()
    return episode_rewards, episode_lengths

def plot_results(rewards, lengths):
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot episode lengths
    plt.subplot(1, 2, 2)
    plt.plot(lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rewards, lengths = train()
    plot_results(rewards, lengths) 