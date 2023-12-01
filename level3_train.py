import gym
from gym import spaces
import numpy as np
import random

class CustomFrozenLakeEnv(gym.Env):
    def __init__(self, size=16, num_holes=10, num_specials=10, start_point=(0, 0), end_point=None):
        assert size >= 4, "Size of the gym should be at least 4x4"
        assert num_holes < size**2 - 2, "Number of holes should be less than total available spaces"

        self.size = size
        self.num_holes = num_holes
        self.num_specials = num_specials
        self.start_point = start_point
        self.end_point = end_point if end_point is not None else (size - 1, size - 1)

        self.observation_space = spaces.Discrete(size**2)
        self.action_space = spaces.Discrete(4)

        self.desc = self.generate_random_environment()
        self.state = self.get_state_from_point(self.start_point)

    def generate_random_environment(self):
        desc = np.full((self.size, self.size), 'F', dtype='<U1')  # 'F' represents frozen surface
        desc[self.start_point] = 'S'  # 'S' represents the starting point
        desc[self.end_point] = 'G'    # 'G' represents the goal

        # Randomly generate holes
        hole_positions = [(i, j) for i in range(self.size) for j in range(self.size)
                          if (i, j) not in [self.start_point, self.end_point]]
        hole_positions = random.sample(hole_positions,self.num_holes)
        # print(len(hole_positions))

        for hole_pos in hole_positions:
            desc[hole_pos] = 'H'  # 'H' represents a hole

        # Randomly generate holes
        special_positions = [(i, j) for i in range(self.size) for j in range(self.size)
                          if (i, j) not in [self.start_point, self.end_point] and (i, j) not in hole_positions]
        special_positions = random.sample(special_positions,self.num_specials)
        # print(len(special_positions))

        for special_pos in special_positions:
            desc[special_pos] = 'R'  # 'H' represents a hole

        return desc

    def get_state_from_point(self, point):
        return point[0] * self.size + point[1]

    def get_point_from_state(self, state):
        return divmod(state, self.size)

    def reset(self):
        self.state = self.get_state_from_point(self.start_point)
        return self.state

    def step(self, action):
        row, col = self.get_point_from_state(self.state)
        if action == 0:  # Move Up
            row = max(0, row - 1)
        elif action == 1:  # Move Down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # Move Left
            col = max(0, col - 1)
        elif action == 3:  # Move Right
            col = min(self.size - 1, col + 1)

        next_state = self.get_state_from_point((row, col))
        reward = 0
        if self.desc[row, col] == 'G': 
            reward = 1  # +1 if the goal is reached
        if self.desc[row, col] == 'R':
            reward = 10  # +10 if the bonus location reached
            self.desc[row, col] = 'F'
        
        done = (self.desc[row, col] == 'H') or (self.desc[row, col] == 'G')  # Done if a hole or the goal is reached

        self.state = next_state
        return next_state, reward, done, {}

    def render(self):
        print("\n".join(["".join(row) for row in self.desc]))
        
    def preprocess_obs(self, obs):
        # Ensure class values are integers within the valid range
        obs = np.clip(obs, 0, self.observation_space.n - 1)

        # One-hot encoding without torch
        obs_one_hot = np.eye(self.observation_space.n)[obs]

        return obs_one_hot


env = CustomFrozenLakeEnv(size=16, num_holes=10, num_specials=10, start_point=(0, 0), end_point=(7, 7))

# Example usage
obs = env.reset()
env.render()


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import time

# ... (unchanged CustomFrozenLakeEnv definition)

# Create the environment
env = CustomFrozenLakeEnv(size=8, num_holes=8, start_point=(0, 0), end_point=(7, 7))
env = DummyVecEnv([lambda: env])  # Wrap the environment for compatibility

# Create the PPO agent
ppo_agent = PPO("MlpPolicy", env, verbose=1)

# Number of training steps
num_steps = 10000

# Train the agent
ppo_agent.learn(total_timesteps=num_steps)

# Save the trained model
ppo_agent.save("level3_custom_frozenlake_model")

# # Evaluate the trained agent
# eval_episodes = 10
# eval_total_rewards = []

# for _ in range(eval_episodes):
#     obs = env.reset()
#     done = False
#     total_reward = 0

#     while not done:
#         action, _ = ppo_agent.predict(obs)
#         obs, reward, done, _ = env.step(action)
#         total_reward += reward
#         env.render()
#         time.sleep(0.5)

#     eval_total_rewards.append(total_reward)

# average_eval_reward = np.mean(eval_total_rewards)
# print(f"\nAverage Evaluation Reward: {average_eval_reward}")
