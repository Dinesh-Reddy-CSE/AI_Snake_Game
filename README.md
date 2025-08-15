# 🐍 AI Snake Game

An **AI-powered Snake Game** built with **Python** and **Pygame**, enhanced with **Deep Reinforcement Learning** to enable autonomous gameplay. The AI learns to navigate, avoid collisions, and optimize its score without human input — demonstrating **AI decision-making**, **neural networks**, and **game programming** concepts in an engaging way.

<p align="center">
  <img src="https://github.com/Dinesh-Reddy-CSE/AI_Snake_Game/blob/main/Demo.gif?raw=true" alt="GIF of AI playing Snake" width="400"/>
</p>

## 🌟 Key Features
-   **Autonomous AI Control:** The game features a fully functional Snake game where the AI plays and learns entirely on its own.
-   **Deep Q-Learning Agent:** The core of the project is an intelligent agent that uses Deep Q-Learning (DQN) to iteratively improve its performance, seeking to maximize its score over thousands of game iterations.
-   **Real-time Rendering:** The game engine, built with Pygame, provides a smooth, real-time visualization of the AI's decision-making process.
-   **Reinforcement Learning Application:** This project is a practical example of how Reinforcement Learning can be applied to solve problems in dynamic, interactive environments.
-   **Modular Python Codebase:** The code is structured into logical components (e.g., game environment, AI agent, neural network model) to make it easy to understand, modify, and extend.

---

## 🛠 Tech Stack
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/Pygame-14354C?logo=pygame&logoColor=white)](https://www.pygame.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-FF6F00?logo=ai&logoColor=white)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Neural Networks](https://img.shields.io/badge/Neural%20Networks-FF0000?logo=tensorflow&logoColor=white)](https://en.wikipedia.org/wiki/Artificial_neural_network)

---

## 📜 How the AI Works: A Deep Dive into Deep Q-Learning
The AI's ability to play the game is powered by a **Deep Q-Learning (DQN)** algorithm. This approach combines a **Q-Learning** agent, which learns from rewards and penalties, with a **neural network** to handle complex states.

### 1. State Representation
The game state is translated into a numerical vector that acts as the input for the neural network. This vector contains crucial information about the snake's environment:
-   **Danger:** Boolean values indicating if there is an imminent collision with a wall or the snake's body in the current direction, or if it turns left or right.
-   **Current Direction:** A one-hot encoded vector representing the snake's current heading (e.g., `[1, 0, 0, 0]` for up).
-   **Food Location:** Boolean values indicating whether the food is above, below, to the left, or to the right of the snake's head.

### 2. Action Space
The AI chooses from three possible actions relative to its current direction. This simplifies the learning problem and prevents illegal 180-degree turns. The actions are:
-   `[1, 0, 0]` → Move **Straight**
-   `[0, 1, 0]` → Turn **Right**
-   `[0, 0, 1]` → Turn **Left**

### 3. Reward Function
The reward function guides the AI's behavior by providing positive feedback for desirable actions and negative feedback for undesirable ones.
-   **+10:** Awarded for eating the food.
-   **-10:** A heavy penalty for a collision with a wall or the snake's own body.
-   **-0.1:** A small penalty for every timestep without progress, encouraging the AI to find the food efficiently.

### 4. Training Method
The agent learns through a process of trial and error, using these key mechanisms:
-   **Experience Replay:** The AI's experiences (state, action, reward, next state, game over) are stored in a memory buffer. The agent then samples random batches from this memory to train its neural network, which makes the learning process more stable.
-   **Epsilon-Greedy Strategy:** During training, the agent balances **exploration** (choosing random actions to discover new strategies) with **exploitation** (choosing the best-known action based on its current knowledge). Over time, the rate of exploration decreases as the agent becomes more confident in its learned policy.
-   **Neural Network:** The network takes the state vector as input and outputs a **Q-value** for each possible action. The Q-value represents the expected future reward for taking that action. The network is continuously trained to predict these values accurately.

---

## 📦 Installation & Running Locally
To get the game running on your machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Dinesh-Reddy-CSE/AI_Snake_Game.git](https://github.com/Dinesh-Reddy-CSE/AI_Snake_Game.git)
    cd AI_Snake_Game
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # On Mac/Linux
    source venv/bin/activate
    # On Windows
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the game locally:**
    This will start the Pygame window, and the AI will begin to play.
    ```bash
    python app.py
    ```

5.  **If running in web mode (using Flask):**
    The web interface will be accessible at `http://localhost:5000` in your browser.
