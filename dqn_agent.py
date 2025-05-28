# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from snake_env import SnakeGame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Neural Network Model ===
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Experience Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# === Epsilon-Greedy Policy ===
def choose_action(model, state, epsilon):
    if np.random.rand() < epsilon:
        return random.randint(0, 2)  # Random action
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = model(state)
        return q_values.argmax().item()

# === Training Loop ===
def train_dqn(episodes=300, batch_size=64, gamma=0.99, eps_decay=0.995, min_epsilon=0.01):
    env = SnakeGame(size=10)
    input_dim = len(env.reset())
    output_dim = 3  # Actions: straight, right, left

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    buffer = ReplayBuffer()
    
    epsilon = 1.0
    best_score = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = choose_action(policy_net, state, epsilon)
            next_state, reward, done = env.step(action)

            buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Learn if there are enough samples
            if len(buffer) > batch_size:
                learn(policy_net, target_net, optimizer, buffer, batch_size, gamma)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * eps_decay)

        # Update target network periodically
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        current_score = len(env.snake)
        print(f"Episode {episode} | Score: {current_score} | Epsilon: {epsilon:.2f}")

        # Save best model
        if current_score > best_score:
            best_score = current_score
            torch.save(policy_net.state_dict(), "best_dqn_snake.pth")

        # Save every 25 episodes
        if episode % 25 == 0 or episode == episodes - 1:
            torch.save(policy_net.state_dict(), f"model_episode_{episode}.pth")

    torch.save(policy_net.state_dict(), "final_dqn_snake.pth")
    print("✅ Training complete.")

# === Learning Function ===
def learn(policy_net, target_net, optimizer, buffer, batch_size, gamma):
    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    current_q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach()
    expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = torch.nn.functional.mse_loss(current_q_values.squeeze(), expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
if __name__ == "__main__":
    train_dqn(episodes=300)