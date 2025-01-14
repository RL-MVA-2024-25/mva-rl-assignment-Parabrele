from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

from pathlib import Path
import numpy as np
import torch as th

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

SAVE_PATH = Path(__file__).parent.parent / "output"
MODEL_NAME = "final.pkl"


class ProjectAgent:
    def act(self, observation, use_random=False):
        observation = np.log(np.maximum(observation, 1e-8))
        return self.policy.predict(observation, deterministic=True)[0]

    def save(self, path):
        th.save(self.policy, path)

    def load(self):
        self.policy = th.load(SAVE_PATH / MODEL_NAME, weights_only=False)