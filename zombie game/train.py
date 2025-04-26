from zombie_env import ZombieEnvironment
from q_learning_agent import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt

def train(episodes=1000, render_every=100):
    # Create environment and agent
    env = ZombieEnvironment()
    agent = QLearningAgent(
        state_size=(env.grid_size, env.grid_size, 4),
        action_size=env.action_space.n
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Choose and perform action
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
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
        
        # Print progress
        if (episode + 1) % render_every == 0:
            avg_reward = np.mean(episode_rewards[-render_every:])
            avg_length = np.mean(episode_lengths[-render_every:])
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Length: {avg_length:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print("--------------------")
    
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