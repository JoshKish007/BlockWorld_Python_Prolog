import gymnasium as gym # running the null agent code as given to see the BlockWorld output.
import aisd_examples
env = gym.make("aisd_examples/BlockWorld-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
