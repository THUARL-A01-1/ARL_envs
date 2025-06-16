import mujoco
from mujoco import viewer
import numpy as np
from RLgrasp.RLgrasp import RLGraspEnv

env = RLGraspEnv()
_ = env.reset()
for i in range(10):
    sample_action = env.action_space.sample()
    # sample_action[0] = 0.0  # r
    # sample_action[1] = 0.0  # beta
    # sample_action[2] = 1.0  # depth_factor
    # sample_action[3] = 0.0  # theta: # 0 ~ pi / 2
    # sample_action[4] = 0.1  # phi: # 0 ~ 2 * pi
    # sample_action[5] = 0.25  # alpha: # 0 ~ 2 * pi
    sample_action[6] = 5.0
    print(f"Sampled action {i+1}: {sample_action}")
    observation, reward, done, truncated, info = env.step(sample_action)
    print(f"reward: {reward}, done: {done}, truncated: {truncated}")
# env.replay()