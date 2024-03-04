import gym
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make('FrozenLake-v1')


num_states = env.observation_space.n
num_actions = env.action_space.n

action_values = np.zeros(shape=(num_states, num_actions))
def policy(state, epsilon =0.2):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))
alpha = 0.1
gamma = 0.99
epsilon= 0.2
episodes = 1000

for i in range(episodes):
    state= env.reset()
    done=False

    action= policy(state,epsilon)
    while not done:
       next_state, reward , done,_ = env.step(action)
       next_action = policy(next_state,epsilon)
       qsa = action_values[state][action]
       next_qsa = action_values[next_state][next_action]
       action_values[state][action] = qsa +alpha*(reward +gamma*next_qsa - qsa)
       state = next_state
       action = next_action

print("Action value")
print(qsa)

def choose_action(state):
    return np.argmax(action_values[state])



total_reward = 0
for i in range(100):
    state = env.reset()
    done = False
    while not done:
        action = choose_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            break

print(f"Average reward over 100 episodes: {total_reward / 100}")
env.close()





