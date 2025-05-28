# visualize_agent.py

import pygame
import numpy as np
import torch
from dqn_agent import DQN
from snake_env import SnakeGame

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize game
env = SnakeGame(size=10)
state_dim = len(env.reset())
action_dim = 3  # [straight, right, left]

# Load model
model = DQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("best_dqn_snake.pth", map_location=device))
model.eval()

# Pygame setup
CELL_SIZE = 40
GRID_SIZE = env.size
SCREEN_SIZE = CELL_SIZE * GRID_SIZE

pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Snake AI Visualization")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 20)

def draw_grid(snake, food):
    screen.fill((0, 0, 0))  # Black background

    # Draw snake
    for segment in snake:
        x, y = segment
        pygame.draw.rect(screen, (0, 255, 0), (y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw head (different color)
    hx, hy = snake[0]
    pygame.draw.rect(screen, (0, 200, 0), (hy*CELL_SIZE, hx*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw food
    fx, fy = food
    pygame.draw.rect(screen, (255, 0, 0), (fy*CELL_SIZE, fx*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Score
    score_text = font.render(f"Score: {len(snake)}", True, (255, 255, 255))
    screen.blit(score_text, (5, 5))

    pygame.display.flip()

# Run visualization
state = env.reset()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Choose action (greedy)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = q_values.argmax().item()

    next_state, reward, done = env.step(action)
    state = next_state

    # Render
    draw_grid(env.snake, env.food)
    clock.tick(10)  # Speed of the game (frames per second)

    if done:
        print(f"Game Over! Final Score: {len(env.snake)}")
        state = env.reset()

pygame.quit()