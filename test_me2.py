import pygame
import gym
from gym import spaces
import time

class Simple2DRaceGame(gym.Env):
    def __init__(self):
        super().__init__()

        # Define action and observation space
        # Actions: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)

        # Observation: current position (2D coordinates)
        self.observation_space = spaces.Tuple((spaces.Discrete(10), spaces.Discrete(10)))  # 10x10 grid

        self.checkpoints = [(1, 1), (3, 1), (5, 5), (6,4), (7, 7)]
        self.start_position = (0, 0)
        self.finish_position = (9, 9)
        self.position = list(self.start_position)

        # Pygame specific setup
        pygame.init()
        self.screen_size = 600
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption('Simple2DRaceGame')
        self.cell_size = self.screen_size // 10  # 10x10 grid

    def reset(self):
        self.position = list(self.start_position)
        return tuple(self.position)

    def step(self, action):
        if action == 0 and self.position[1] > 0:  # move up
            self.position[1] -= 1
        elif action == 1 and self.position[1] < 9:  # move down
            self.position[1] += 1
        elif action == 2 and self.position[0] > 0:  # move left
            self.position[0] -= 1
        elif action == 3 and self.position[0] < 9:  # move right
            self.position[0] += 1

        reward = 0
        if tuple(self.position) in self.checkpoints:  # checkpoint
            reward = 10
        elif tuple(self.position) == self.finish_position:  # finish line
            reward = 100

        done = tuple(self.position) == self.finish_position
        return tuple(self.position), reward, done, {}

    def render(self, mode='human', option_target=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Colors
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)

        self.screen.fill(WHITE)

        # Draw agent
        pygame.draw.circle(self.screen, RED, (int((self.position[0] + 0.5) * self.cell_size), int((self.position[1] + 0.5) * self.cell_size)), 20)

        # Draw checkpoints
        for cp in self.checkpoints:
            pygame.draw.rect(self.screen, BLUE, (cp[0] * self.cell_size, cp[1] * self.cell_size, self.cell_size, self.cell_size), 2)

        # Draw finish line
        pygame.draw.rect(self.screen, GREEN, (self.finish_position[0] * self.cell_size, self.finish_position[1] * self.cell_size, self.cell_size, self.cell_size))

        # Draw option target
        if option_target:
            pygame.draw.circle(self.screen, GREEN, (int((option_target[0] + 0.5) * self.cell_size), int((option_target[1] + 0.5) * self.cell_size)), 25, 2)

        pygame.display.flip()

    def close(self):
        pygame.quit()



import numpy as np

class MetaController:
    def __init__(self, n_options):
        self.n_options = n_options
        self.value_table = np.zeros(n_options)

    def select_option(self):
        # Here, we use a simple epsilon-greedy strategy for option selection
        epsilon = 0.1
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_options)
        else:
            return np.argmax(self.value_table)

    def update(self, option, reward):
        alpha = 0.1
        self.value_table[option] += alpha * (reward - self.value_table[option])

class Option:
    def __init__(self, target_position):
        self.target_position = target_position
        self.value_table = np.zeros((10, 10))  # Change this to a 2D array for 10x10 grid

    def act(self, current_position):
        dx = self.target_position[0] - current_position[0]
        dy = self.target_position[1] - current_position[1]

        # If the target is to the right
        if dx > 0:
            return 3
        # If the target is to the left
        elif dx < 0:
            return 2
        # If the target is below
        elif dy > 0:
            return 1
        # If the target is above
        elif dy < 0:
            return 0
        # If already at the target
        else:
            return np.random.choice(4)  # choose a random action

    def update(self, current_position, reward):
        alpha = 0.1
        x, y = current_position
        self.value_table[x][y] += alpha * (reward - self.value_table[x][y])


# Now, integrate HRL with the environment
env = Simple2DRaceGame()

meta_controller = MetaController(n_options=len(env.checkpoints) + 1)
options = [Option(cp) for cp in env.checkpoints] + [Option(env.finish_position)]


for episode in range(10000):  # Train for 100 episodes
    state = env.reset()
    total_reward = 0
    while True:
        option_idx = meta_controller.select_option()
        option = options[option_idx]
        cumulated_option_reward = 0
        while state != option.target_position:
            env.render(option_target=option.target_position)
            action = option.act(state)
            next_state, reward, done, _ = env.step(action)
            option.update(state, reward)
            cumulated_option_reward += reward
            state = next_state
            time.sleep(0.1)
            if done:
                break
        meta_controller.update(option_idx, cumulated_option_reward)
        total_reward += cumulated_option_reward
        if done:
            break
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

