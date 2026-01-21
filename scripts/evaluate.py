import argparse
import gym
import torch
import numpy as np
from train import AtariPreprocessor, QNetwork
import sys

def evaluate_agent(game='Pong-v5', episodes=100, model_path='models/best_model.pth'):
    env = gym.make(game)
    preprocessor = AtariPreprocessor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    q_network = QNetwork(env.action_space.n).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()
    
    total_rewards = []
    for episode in range(episodes):
        state = preprocessor.reset(env.reset()[0])
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
            
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocessor.step(next_state_raw)
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        if (episode + 1) % 20 == 0:
            print(f'Episode {episode+1}/{episodes}, Reward: {episode_reward}')
    
    avg_reward = np.mean(total_rewards)
    print(f'AVERAGE_REWARD: {avg_reward:.2f}')
    sys.stdout.flush()
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Pong-v5')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--model-path', default='models/best_model.pth')
    args = parser.parse_args()
    
    evaluate_agent(args.game, args.episodes, args.model_path)
