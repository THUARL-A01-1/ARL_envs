import gymnasium as gym
import os
from RLgrasp.RLgrasp import RLGraspEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

# 并行环境数量
NUM_ENVS = 1

def make_env():
    def _init():
        # 替换为你的自定义环境
        env = RLGraspEnv(render_mode="human")
        return env
    return _init

if __name__ == "__main__":
    # 创建多线程向量化环境
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # 训练配置
    config = {
        "learning_rate": 3e-4,             # 学习率
        "batch_size": 16,                  # 批量大小
        "n_steps": 32,                   # 每次更新的步数（PPO等）
        "gamma": 0.99,                     # 折扣因子
        "gae_lambda": 0.95,                # GAE参数
        "clip_range": 0.2,                 # PPO裁剪范围
        "ent_coef": 0.0,                   # 熵系数
        "vf_coef": 0.5,                    # 值函数损失系数
        "max_grad_norm": 0.5,              # 最大梯度范数
        "policy_kwargs": dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # 策略和价值网络结构
            activation_fn= torch.nn.ReLU,  # 激活函数
            normalize_images=False,
        ),
          # 是否归一化图像
        "verbose": 1,                   # 日志级别
        # "env_kwargs": dict(
        #     depth_size=84,                 # 深度图尺寸
        #     history_len=4,                 # 历史帧数
        #     reward_type="dense",           # 奖励类型
        # ),
        # "num_envs": 8,                     # 并行环境数
        # "save_freq": 10_000,               # 保存频率
        # "eval_freq": 50_000,               # 验证频率
        # "seed": 42,                        # 随机种子
    }
    

    # 创建模型
    model = PPO("CnnPolicy", env, **{k: v for k, v in config.items()})
    
    # 训练
    model.learn(total_timesteps=1000)

    # 保存模型
    model.save("RLgrasp/logs/ppo")

    # 关闭环境
    env.close()