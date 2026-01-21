import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

class AtariPreprocessor:
    def __init__(self, frame_size=84, frame_skip=4):
        self.frame_size = frame_size
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=4)
        
    def preprocess(self, frame):
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.frame_size, self.frame_size))
        return resized
    
    def reset(self, frame):
        preprocessed = self.preprocess(frame)
        for _ in range(4):
            self.frames.append(preprocessed)
        return self.get_state()
    
    def step(self, frame):
        preprocessed = self.preprocess(frame)
        self.frames.append(preprocessed)
        return self.get_state()
    
    def get_state(self):
        return np.stack(list(self.frames), axis=0)

class QNetwork(nn.Module):
    def __init__(self, action_size=4):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, action_size, lr=0.0001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = 0.995
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.q_network = QNetwork(action_size).to(self.device)
        self.target_network = QNetwork(action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def train_step(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.functional.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Pong-v5')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.01)
    args = parser.parse_args()
    
    env = gym.make(args.game)
    preprocessor = AtariPreprocessor()
    agent = DQNAgent(env.action_space.n, lr=args.learning_rate, gamma=args.gamma, 
                     epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end)
    
    os.makedirs('models', exist_ok=True)
    
    for episode in range(args.episodes):
        state = preprocessor.reset(env.reset()[0])
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocessor.step(next_state_raw)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train_step(args.batch_size)
            state = next_state
            episode_reward += reward
        
        if (episode + 1) % 10 == 0:
            agent.update_target_network()
        
        agent.decay_epsilon()
        
        if (episode + 1) % 20 == 0:
            torch.save(agent.q_network.state_dict(), f'models/model_episode_{episode+1}.pth')
            print(f'Episode {episode+1}/{args.episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}')
    
    torch.save(agent.q_network.state_dict(), 'models/best_model.pth')
    print('Training complete. Model saved as best_model.pth')
    env.close()

if __name__ == '__main__':
    main()
