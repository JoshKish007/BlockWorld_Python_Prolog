from gymnasium.envs.registration import register

register(
    id="aisd_examples/BlockWorld-v0",
    entry_point="aisd_examples.envs:BlockWorldEnv",
)
