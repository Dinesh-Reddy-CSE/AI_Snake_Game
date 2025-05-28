# snake_env.py

import numpy as np
import random

class SnakeGame:
    def __init__(self, size=10, mode='normal'):
        self.size = size
        self.snake = []
        self.direction = ''
        self.food = ()
        self.mode = mode  # Add mode support
        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.direction = 'right'
        self.spawn_food()
        return self.get_state()

    def spawn_food(self):
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dx = food_x - head_x
        dy = food_y - head_y

        dir_left = self.direction == 'left'
        dir_right = self.direction == 'right'
        dir_up = self.direction == 'up'
        dir_down = self.direction == 'down'

        danger_left = ((head_x - 1, head_y) in self.snake) or head_x - 1 < 0
        danger_right = ((head_x + 1, head_y) in self.snake) or head_x + 1 >= self.size
        danger_up = ((head_x, head_y - 1) in self.snake) or head_y - 1 < 0
        danger_down = ((head_x, head_y + 1) in self.snake) or head_y + 1 >= self.size

        state = [
            danger_left, danger_right, danger_up, danger_down,
            dx < 0, dx > 0, dy < 0, dy > 0,
            dir_left, dir_right, dir_up, dir_down
        ]

        return np.array(state, dtype=int)

    def step(self, action):
        dirs = ['up', 'right', 'down', 'left']
        idx = dirs.index(self.direction)

        if action == 1:  # Right turn
            self.direction = dirs[(idx + 1) % 4]
        elif action == 2:  # Left turn
            self.direction = dirs[(idx - 1) % 4]

        x, y = self.snake[0]
        if self.direction == 'up':
            x -= 1
        elif self.direction == 'down':
            x += 1
        elif self.direction == 'left':
            y -= 1
        elif self.direction == 'right':
            y += 1

        new_head = (x, y)

        done = False
        reward = 0

        if (
            x < 0 or x >= self.size or
            y < 0 or y >= self.size or
            new_head in self.snake
        ):
            done = True
            reward = -10
            return self.get_state(), reward, done

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.spawn_food()
            reward = 10
        else:
            self.snake.pop()  # move forward

        return self.get_state(), reward, done