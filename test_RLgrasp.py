import mujoco
from mujoco import viewer
import numpy as np
from RLgrasp.RLgrasp import RLGraspEnv

env = RLGraspEnv()
for i in range(10):
    _ = env.reset()
    sample_action = env.action_space.sample()
    print(f"Sampled action {i+1}: {sample_action}")
    env.step(sample_action)
# env.replay()