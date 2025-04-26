# Zombie Game - Reinforcement Learning Project

A reinforcement learning project where an agent learns to navigate a maze, defeat zombies in the correct order, and reach the exit.

## Project Structure

- `zombie_env_short.py`: The game environment implementation
- `q_learning_agent.py`: Q-learning agent implementation
- `train_q_learning.py`: Training script
- `assets/`: Directory containing game sprites
- `q_table.npy`: Saved Q-table from training

## Environment Details

The game environment features:
- 8x8 grid world
- Fixed positions for all entities:
  - Warrior (player) starts in upper left
  - Level 1 zombie in upper right
  - Level 10 zombie in lower right
  - Level 100 zombie in lower left
  - Exit door in the middle
- Maze-like wall structure
- Fast movement speed for quick training

## Requirements

- Python 3.x
- Required packages:
  - numpy
  - pygame
  - gymnasium

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Agent

To train the Q-learning agent:
```bash
python train_q_learning.py
```

Training parameters:
- 5000 maximum episodes
- Early stopping if reward > 5000
- Progress updates every 5 episodes
- Model saved when new best reward achieved
- 200 steps maximum per episode

## How to Play

1. Run the training script
2. Watch the agent learn to:
   - Navigate the maze
   - Defeat zombies in order (L1 → L10 → L100)
   - Reach the exit
3. The agent's performance improves over time as it learns optimal strategies

## Key Features

- Q-learning implementation
- Fixed entity positions for consistent learning
- Maze-like wall structure
- Fast movement speed
- Visual rendering of the game
- Progress tracking and model saving

## Notes

- The agent learns through trial and error
- Rewards are given for:
  - Moving towards target zombie (+5)
  - Moving towards exit (+10)
  - Killing zombies (L1: +20, L10: +200, L100: +2000)
  - Killing all zombies (+500)
  - Reaching exit (+5000)
  - Penalties for hitting walls (-1) and wrong zombie order (-200) 