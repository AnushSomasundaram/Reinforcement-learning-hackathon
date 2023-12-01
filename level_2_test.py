import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from level2 import CustomFrozenLakeEnv

# Set your environment name
environment_name = 'FrozenLake-v1'
render_mode = 'human'

# Create a custom FrozenLake environment
custom_lake = CustomFrozenLakeEnv(size=8, num_holes=8, start_point=(0, 0), end_point=(7, 7))

# ------------------------------------------------------------------------------------------
# training on random board that has been generated


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import time

# ... (unchanged CustomFrozenLakeEnv definition)

# # Create the environment
env = custom_lake
env = DummyVecEnv([lambda: env])  # Wrap the environment for compatibility

# # Create the PPO agent
ppo_agent = PPO("MlpPolicy", env, verbose=1)

# # Number of training steps
num_steps = 1000000

# # Train the agent
ppo_agent.learn(total_timesteps=num_steps)

# # Save the trained model
model_save_path = "/Users/software/Desktop/reinforcement_learning_practise/hackathon/"
os.makedirs(model_save_path, exist_ok=True)
location = os.path.join(model_save_path, "level2_custom_frozenlake_model")
ppo_agent.save(location)




# ------------------------------------------------------------------------------------------


# trial runs

custom_lake.render()
custom_lake = [''.join(sublist) for sublist in custom_lake.desc]

print(custom_lake)
# Attempt to create a Gymnasium environment
try:
    gym_env = gym.make(environment_name, desc=custom_lake, render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: gym_env])
except gym.error.UnregisteredEnv:
    # If the Gymnasium environment is not available, use the OpenAI Gym environment
    vec_env = DummyVecEnv([lambda: gym.make(environment_name, desc=custom_lake, render_mode=render_mode)])





# Load the trained model
model = PPO.load(location)

# Set the number of episodes for the trial
num_episodes = 50

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