import gymnasium as gym # running the null agent code as given to see the BlockWorld output.
import aisd_examples
env = gym.make("aisd_examples/BlockWorld-v0", render_mode="human")
observation, info = env.reset() # resetting the environment to its initial state.

for _ in range(1000): # running 1000 iterations
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)  # the new observation, reward, termination status,
    # truncation status, and info dictionary are shown after each execution.

    if terminated or truncated: # if condition loop checking the episode has terminated or not.
        observation, info = env.reset() # starting a new episode.

env.close()
