import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Set your environment name
environment_name = 'FrozenLake-v1'
#render_mode = 'human'

# Create the environment
env = gym.make(environment_name)
# Wrap the environment in DummyVecEnv for compatibility with Stable Baselines
vec_env = DummyVecEnv([lambda: env])

# Set hyperparameters
total_timesteps = 100000
learning_rate = 0.001
batch_size = 64

# Create the model with specified hyperparameters
model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=learning_rate, batch_size=batch_size)

# ------------------------------
# training script

# Display training info
model.learn(total_timesteps=total_timesteps)
model.save("trained_level_1_frozen_lake")
# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

