import gym
import numpy as np
from gym import spaces
from unityagents import UnityEnvironment


class ReacherEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env_filepath):
        super(ReacherEnv, self).__init__()

        self.unity_env = UnityEnvironment(file_name=env_filepath)
        # get the default brain
        self.brain_name = self.unity_env.brain_names[0]
        brain = self.unity_env.brains[self.brain_name]

        env_info = self.unity_env.reset(train_mode=True)[self.brain_name]

        state = env_info.vector_observations[0]
        self.state_size = len(state)
        self.action_size = brain.vector_action_space_size
        self.num_agents = len(env_info.agents)

        high = np.ones(self.action_size)
        self.action_space = spaces.Box(low=-high, high=high, dtype=np.float)

        high = np.ones(self.state_size)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float)

    def step(self, action):
        env_info = self.unity_env.step(action)[self.brain_name]

        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        obs = env_info.vector_observations[0]

        return obs, reward, done, {}

    def reset(self, train_mode=True):
        env_info = self.unity_env.reset(train_mode=train_mode)[self.brain_name]
        return env_info.vector_observations[0]

    def render(self, mode='human'):
        pass

    def close(self):
        self.unity_env.close()
