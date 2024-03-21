import numpy as np
import time
import os
import matplotlib.pyplot as plt
import gymnasium as gym
import aisd_examples
env = gym.make("aisd_examples/BlockWorld-v0", render_mode="human")

# Calculate the number of states
num_states = len(env.states_dict)

# Create Q-table with random values
qtable = np.random.rand(num_states, env.action_space.n).tolist()

# Hyperparameters
episodes = 20
gamma = 0.1
epsilon = 0.06
decay = 0.2

# Lists to store episode returns and steps per episode
episode_returns = []
steps_per_episode = []

# Training loop
for i in range(episodes):
    state_box, info = env.reset()
    state = state_box['agent']
    steps = 0
    episode_return = 0
    done = False

    while not done:
        os.system('clear')
        print("episode #", i+1, "/", episodes)
        env.render()
        time.sleep(0.05)

        # Increment steps
        steps += 1

        # Exploration-exploitation trade-off
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = qtable[state].index(max(qtable[state]))

        # Take action
        next_state_box, reward, done, _, info = env.step(action)
        next_state = next_state_box['agent']

        # Update Q-table using Bellman equation
        qtable[state][action] = reward + gamma * max(qtable[next_state])

        # Update state and episode return
        state = next_state
        episode_return += reward

    # Decay epsilon
    epsilon -= decay * epsilon

    # Append episode return and steps per episode to lists
    episode_returns.append(episode_return)
    steps_per_episode.append(steps)

    print("\nDone in", steps, "steps")
    print("\nEpisode return:", episode_return)


# Plot episode returns and steps per episode
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(range(1, episodes + 1), episode_returns)
plt.title('Episode Returns Hyperparameter set 8')
plt.xlabel('Episode')
plt.ylabel('Return')

plt.subplot(2, 1, 2)
plt.plot(range(1, episodes + 1), steps_per_episode)
plt.title('Steps per Episode Hyperparameter set 8')
plt.xlabel('Episode')
plt.ylabel('Steps')

plt.tight_layout()
plt.savefig('episode_returns_and_steps.png')

env.close()
