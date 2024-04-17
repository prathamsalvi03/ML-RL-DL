import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from gym.wrappers import ObservationWrapper  # Import the ObservationWrapper class

# Define a custom observation wrapper to convert observations to grayscale
class GrayscaleObservation(ObservationWrapper):
    def __init__(self, env):
        super(GrayscaleObservation, self).__init__(env)
        self.observation_space = env.observation_space  # Keep the same observation space

    def observation(self, observation):
        # Convert the RGB observation to grayscale
        observation = np.dot(observation[..., :3], [0.2989, 0.5870, 0.1140])
        return observation

# Create the Super Mario Bros environment
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
if 'render_modes' not in env.metadata or env.metadata['render_modes'] is None:
    print("Warning: No render modes declared in the environment. You may encounter issues when calling .render().")

# Apply the JoypadSpace wrapper to map discrete actions to the NES controller
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Apply the custom grayscale observation wrapper
env = GrayscaleObservation(env)

# Create a DummyVecEnv and apply VecFrameStack wrapper
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')  # 4 is the number of frame stack

# Run the environment
done = True
for step in range(5000):
    if done:
        # Reset the environment and get the initial observation
        state = env.reset()
    # Sample a random action from the environment's action space
    action = env.action_space.sample()
    # Take a step in the environment with the sampled action
    state, reward, done, info = env.step([action])
    # Render the environment
    env.render()

# Close the environment
env.close()

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TLCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TLCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TLCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

model = PPO('CnnPolicy', env, verbose=1, learning_rate=0.000001, n_steps=512)

model.learn(total_timesteps=10000, callback=callback)
