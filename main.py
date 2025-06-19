import gymnasium as gym
import os
from RLgrasp.RLgrasp import RLGraspEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import NatureCNN
import torch
import torch.nn as nn

def make_env(mode="train"):
    def _init():
        if mode == "train":
            env = RLGraspEnv(render_mode="rgb_array", grasp_mode="fixed_force", scene_range=range(50))  # 每个进程独立初始化
        if mode == "eval":
            env = RLGraspEnv(render_mode="rgb_array", grasp_mode="fixed_force", scene_range=range(50, 88))
        return Monitor(env)
    return _init

class CustomCNN(NatureCNN):
    def __init__(self, *args, **kwargs):
        super(CustomCNN, self).__init__(*args, **kwargs)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            )
        self.linear = nn.Sequential(nn.Linear(1600, 64), nn.ReLU())

if __name__ == "__main__":
    # 训练配置
    config = {
        "learning_rate": 3e-4,             # 学习率
        "batch_size": 16,                  # 批量大小
        "n_steps": 10,                   # 每次更新的步数（PPO等）
        "gamma": 0.99,                     # 折扣因子
        "gae_lambda": 0.95,                # GAE参数
        "clip_range": 0.2,                 # PPO裁剪范围
        "ent_coef": 0.0,                   # 熵系数
        "vf_coef": 0.5,                    # 值函数损失系数
        "max_grad_norm": 0.5,              # 最大梯度范数
        "policy_kwargs": dict(
            features_extractor_class=CustomCNN,  # 自定义特征提取器
            features_extractor_kwargs=dict(features_dim=64),  # 特征提取器参数
            net_arch=dict(pi=[64, 64], vf=[64, 64]),  # 策略和价值网络结构
            activation_fn= nn.ReLU,  # 激活函数
            normalize_images=False,        # 是否归一化图像
        ),

        "verbose": 1,                   # 日志级别
        "tensorboard_log": "RLgrasp/logs/ppo_test",  # TensorBoard日志目录
        # "env_kwargs": dict(
        #     depth_size=84,                 # 深度图尺寸
        #     history_len=4,                 # 历史帧数
        #     reward_type="dense",           # 奖励类型
        # ),
        "train_envs": 16,                     # 并行环境数
        "eval_envs": 2,
        # "save_freq": 10_000,               # 保存频率
        "eval_freq": 40,               # 验证频率
        # "seed": 42,                        # 随机种子
    }

    # 创建多线程向量化环境
    train_env = SubprocVecEnv([make_env("train") for _ in range(config["train_envs"])])
    eval_env = SubprocVecEnv([make_env("eval") for _ in range(config["eval_envs"])])
    # env = DummyVecEnv([make_env()])
    # eval_env = DummyVecEnv([make_env()])

    # 评估回调
    eval_callback = EvalCallback(train_env, best_model_save_path=config["tensorboard_log"], log_path=config["tensorboard_log"], eval_freq=config['eval_freq'])

    # 创建模型
    model = PPO("CnnPolicy", train_env, **{k: v for k, v in config.items() if k not in ["train_envs", "eval_envs", "eval_freq"]})
    
    # 训练
    model.learn(total_timesteps=100000, callback=eval_callback)

    # 保存模型
    model.save(config["tensorboard_log/final_model"])

    # 关闭环境
    train_env.close()
    eval_env.close()