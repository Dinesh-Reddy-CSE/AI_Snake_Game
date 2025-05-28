# fast_ai_player.py

from snake_env import SnakeGame
from dqn_agent import DQN
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = 12
output_dim = 3
current_model_path = "best_dqn_snake.pth"
env = SnakeGame(size=10)  # No 'mode' passed unless needed
model = None

def load_model(model_path="best_dqn_snake.pth"):
    global model
    model = DQN(input_dim, output_dim).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    model.eval()

load_model(current_model_path)

state = env.reset()

def set_model(model_path="best_dqn_snake.pth", mode="normal"):
    global env, state
    env = SnakeGame(size=10)
    state = env.reset()
    load_model(model_path)

def get_next_move(state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
        return q_values.argmax().item()

def get_ai_game_frame():
    global state, env
    while True:
        action = get_next_move(state)
        next_state, reward, done = env.step(action)
        state = next_state

        yield {
            "snake": env.snake,
            "food": env.food,
            "score": len(env.snake),
            "done": done
        }

        if done:
            state = env.reset()