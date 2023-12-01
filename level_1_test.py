import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Set your environment name
environment_name = 'FrozenLake-v1'
render_mode = 'human'

# Attempt to create a Gymnasium environment
try:
    gym_env = gym.make(environment_name, render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: gym_env])
except gym.error.UnregisteredEnv:
    # If the Gymnasium environment is not available, use the OpenAI Gym environment
    vec_env = DummyVecEnv([lambda: gym.make(environment_name, render_mode=render_mode)])

# Load the trained model
model = PPO.load("/Users/software/Desktop/reinforcement_learning_practise/trained_level_1_frozen_lake.zip")

# Set the number of episodes for the trial
num_episodes = 10

# Run a trial of various episodes
for episode in range(num_episodes):
    obs = vec_env.reset()
    total_reward = 0
    done = False
    episode_path = {"observations": [], "actions": [], "rewards": []}

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        # Store observations, actions, and rewards during the episode
        episode_path["observations"].append(obs.copy())
        episode_path["actions"].append(action)
        episode_path["rewards"].append(reward)

        total_reward += reward

    print(f"Episode {episode + 1} - Total Reward: {total_reward}")

    # Check if the episode was successful
    if total_reward == 1:
        print("Episode succeeded!")
        print("Observations:", episode_path["observations"])
        print("Actions:", episode_path["actions"])
        print("Rewards:", episode_path["rewards"])

# Close the environment
vec_env.close()
