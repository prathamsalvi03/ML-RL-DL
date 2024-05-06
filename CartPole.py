import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, num_hidden):
        super(ActorCritic, self).__init__()
        self.common = nn.Linear(num_inputs, num_hidden)
        self.actor = nn.Linear(num_hidden, num_actions)
        self.critic = nn.Linear(num_hidden, 1)

    def forward(self, inputs):
        common = torch.relu(self.common(inputs))
        action_probs = torch.softmax(self.actor(common), dim=-1)
        value_estimate = self.critic(common)
        return action_probs, value_estimate

def discounted_rewards(rewards, gamma=0.99):
    discounted = []
    running_add = 0
    for r in reversed(rewards):
        running_add = r + gamma * running_add
        discounted.append(running_add)
    return list(reversed(discounted))

def run_episode(model, env, max_steps_per_episode=10000, render=False):
    states, actions, probs, rewards, critic = [], [], [], [], []
    state = env.reset()
    for _ in range(max_steps_per_episode):
        if render:
            env.render()

        # Get action probabilities and critic value from the model
        action_probs, critic_value = model(torch.tensor(state, dtype=torch.float32))

        # Sample an action from the action probabilities
        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())

        # Append state, action, action probability, and critic value
        states.append(state)
        actions.append(action)
        probs.append(torch.log(action_probs[action]))
        critic.append(critic_value.item())

        # Take a step in the environment
        state, reward, done, _ = env.step(action)  # Update state with the next state from the environment
        if done:
            break

        # Append reward
        rewards.append(reward)

    return states, actions, probs, rewards, critic

env = gym.make("CartPole-v1")
print(f"Action space: {env.action_space}")

num_inputs = 4
num_actions = 2
num_hidden = 128

model = ActorCritic(num_inputs, num_actions, num_hidden)

optimizer = optim.Adam(model.parameters(), lr=0.02)
huber_loss = nn.SmoothL1Loss()
episode_count = 0
running_reward = 0

while True:
    state = env.reset()
    episode_reward = 0
    action_probs, _, rewards, critic_value, _ = run_episode(model, env)
    episode_reward = sum(rewards)
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
    dr = discounted_rewards(rewards)
    actor_losses = []
    critic_losses = []

    for log_prob, value, rew in zip(action_probs, critic_value, dr):
        diff = rew - value.item()  # Convert value tensor to scalar
        actor_losses.append(-log_prob * diff)
        critic_losses.append(huber_loss(value.unsqueeze(0), torch.tensor([rew])))  # Apply Huber loss

    # Backpropagation
    loss_value = sum(actor_losses) + sum(critic_losses)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

env.close()  # Close the environment after training

# Visualize the trained model's performance
_ = run_episode(model, env, render=True)  # Only call this once

env.close() 
