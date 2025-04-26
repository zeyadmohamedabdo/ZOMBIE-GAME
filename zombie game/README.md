# Zombie Soldier Reinforcement Learning Project

This project implements a Q-learning agent that learns to navigate a grid world as a soldier fighting zombies. The soldier must defeat zombies in a specific order (level 1, then level 10, then level 100) to win the game.

## Environment Description

- The environment is a grid world where the soldier (agent) must navigate and defeat zombies
- There are three zombies with different levels (1, 10, 100)
- The soldier must defeat the zombies in ascending order of their levels
- Attempting to attack a higher-level zombie before defeating lower-level zombies results in a penalty

### Actions
- 0: Move Up
- 1: Move Right
- 2: Move Down
- 3: Move Left
- 4: Attack

### Rewards
- Defeating level 1 zombie: +1
- Defeating level 10 zombie: +10
- Defeating level 100 zombie: +100
- Winning the game (defeating all zombies in order): +1000
- Invalid attack (wrong order): -50
- Each step: -0.1

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Running the Project

To train the agent:
```bash
python train.py
```

The training script will:
1. Create the environment and Q-learning agent
2. Train for 1000 episodes by default
3. Display progress every 100 episodes
4. Show a plot of rewards and episode lengths at the end

## Visualization

The environment uses Pygame for visualization:
- Green circle: Player (soldier)
- Gray square: Level 1 zombie
- Dark red square: Level 10 zombie
- Bright red square: Level 100 zombie

## Project Structure

- `zombie_env.py`: Contains the main environment implementation
- `q_learning_agent.py`: Implements the Q-learning agent
- `train.py`: Main script for training the agent
- `requirements.txt`: List of required Python packages 