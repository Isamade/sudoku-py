import torch
from env import SudokuEnv
from rl_agent import DQNAgent
import numpy as np

def train_rl():
    env = SudokuEnv()
    num_episodes = 2000
    batch_size = 64
    target_update_freq = 10
    
    # State: 16 values, Actions: 16 cells * 4 values = 64
    agent = DQNAgent(state_dim=16, action_dim=64)
    
    print("Starting RL training for Sudoku solver...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action_idx = agent.select_action(state)
            
            # Convert action_idx (0-63) to (cell_idx 0-15, value 1-4)
            cell_idx = action_idx // 4
            val = (action_idx % 4) + 1
            
            next_state, reward, done, info = env.step((cell_idx, val))
            
            agent.memory.append((state, action_idx, reward, next_state, done))
            agent.train(batch_size)
            
            state = next_state
            episode_reward += reward
            
        if (episode + 1) % target_update_freq == 0:
            agent.update_target()
            
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Save the model
    torch.save(agent.model.state_dict(), "sudoku_rl.pth")
    print("RL Model saved to sudoku_rl.pth")

if __name__ == "__main__":
    train_rl()
