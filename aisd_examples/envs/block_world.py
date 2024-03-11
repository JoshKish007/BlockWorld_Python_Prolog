import gymnasium as gym
from gymnasium import spaces
import pygame
from screen import Display
from swiplserver import PrologMQI, PrologThread
import numpy as np

# Define a custom Gym environment for a block world
class BlockWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        # Initialize the environment
        self.render_mode = render_mode

        # Create PrologMQI instance and PrologThread
        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()

        # Set up dictionary to convert Prolog state names into integers    
        states_result = self.prolog_thread.query("[blocks_world]")    
        states_result = self.prolog_thread.query("state(State)")
        self.states_dict = {}
        for i, state in enumerate(states_result):
            state_string = state['State']
            self.states_dict[state_string] = i
        
        # Define the observation space
        self.observation_space = spaces.Dict({
                "agent": spaces.Discrete(len(self.states_dict)),
                "target": spaces.Discrete(len(self.states_dict))
            }
        )

        # Set up dictionary to convert action numbers into Prolog actions
        self.actions_dict = {}
        actions_result = self.prolog_thread.query("action(A)")
        for i, A in enumerate(actions_result):
            action_string = A['A']['functor']
            first = True
            for arg in A['A']['args']:
                if first:
                    first = False
                    action_string += '('
                else:
                    action_string += ','
                action_string += str(arg)
            action_string += ')'
            self.actions_dict[i] = action_string

        # Define the action space
        self.action_space = spaces.Discrete(len(self.actions_dict))

        # Initialize state values
        state_integer = np.random.randint(0, len(self.states_dict), size=1, dtype=int)
        state_str = list(self.states_dict.keys())[list(self.states_dict.values()).index(state_integer)]
        self._agent_location = state_integer
        
        # Initialize display if render mode is set to "human"
        if self.render_mode == "human":
            self.display = Display()
            
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    # Helper function to get observations
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    # Helper function to get additional information
    def _get_info(self):
        return {
            "distance": abs(
                self._agent_location - self._target_location
            )
        }
    
    # Reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.display is not None:
            # Randomly choose an initial state
            state_integer = np.random.randint(0, len(self.states_dict), size=1, dtype=int)
            self.state = state_integer
            state_value = list(self.states_dict.keys())[list(self.states_dict.values()).index(state_integer)]
            
            # Extract target and agent values from the state
            target_value = state_value[-3:]
            agent_value = state_value[0:3]
            
            # Set agent location and display target
            self._agent_location = state_integer
            self.display.target = target_value
                       
            # Store initial state values
            self.start_value = state_value
            self.target_value = target_value
            self.agent_value = agent_value

        # Issue Prolog query to reset
        self.prolog_thread.query("reset")

        # Issue Prolog query to retrieve current state
        result = self.prolog_thread.query("current_state(State)")
        agent_state = result[0]['State']
        
        # Update agent and target locations based on Prolog state
        self._agent_location = list(self.states_dict.values())[list(self.states_dict.keys()).index(agent_state)]
        self._target_location = self._agent_location
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()

        return observation, info

    # Perform a step in the environment given an action
    def step(self, action):
        # Convert action number to Prolog action string
        act_str = list(self.actions_dict.values())[list(self.actions_dict.keys()).index(action)]
        query_str = 'step('+act_str+')'
        
        # Issue Prolog query to perform the action
        result = self.prolog_thread.query(query_str)
        
        reward = 0
        terminated = False
        
        if result:
            # If the action is successful, update the agent state and provide a negative reward
            current_state = self.prolog_thread.query("current_state(State)")
            self.agent_value = current_state[0]['State']
            reward = -1
            
            if self.agent_value == self.display.target:
                # If the agent reaches the target state, provide a positive reward and terminate the episode
                reward = 100
                terminated = True
                
        else:
            # If the action fails, provide a negative reward
            reward = -10
            
        if self.render_mode == "human":
            self.render()
            
        # Update the agent location based on the Prolog state
        self._agent_location = list(self.states_dict.values())[list(self.states_dict.keys()).index(self.agent_value)]

        observations = self._get_obs()
        return observations, reward, terminated, False, {}

    # Render the environment
    def render(self):
        if self.render_mode == "human":
            self.display.step(self.agent_value)

    # Close the environment
    def close(self):
        if self.mqi is not None:
            # Stop the PrologMQI instance
            self.mqi.stop()
