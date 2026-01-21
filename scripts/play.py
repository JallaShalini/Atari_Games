import argparse
import gym
import torch
import numpy as np
from train import AtariPreprocessor, QNetwork
import os

def play_agent(game='Pong-v5', episodes=5, model_path='models/best_model.pth', output_dir='videos'):
    os.makedirs(output_dir, exist_ok=True)
    
    env = gym.make(game, render_mode='rgb_array')
    preprocessor = AtariPreprocessor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    q_network = QNetwork(env.action_space.n).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()
    
    for episode in range(episodes):
        frames = []
        state = preprocessor.reset(env.reset()[0])
        episode_reward = 0
        done = False
        
        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
            
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocessor.step(next_state_raw)
            state = next_state
            episode_reward += reward
        
        if frames:
            try:
                import imageio
                output_path = f'{output_dir}/gameplay_episode_{episode+1}.mp4'
                imageio.mimsave(output_path, frames, fps=30)
                print(f'Saved gameplay video to {output_path}')
            except ImportError:
                print('imageio not available, skipping video save')
        
        print(f'Episode {episode+1}/{episodes}, Reward: {episode_reward}')
    
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Pong-v5')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--model-path', default='models/best_model.pth')
    parser.add_argument('--output-dir', default='videos')
    args = parser.parse_args()
    
    play_agent(args.game, args.episodes, args.model_path, args.output_dir)
