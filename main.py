import gymnasium as gym
import os
from RLgrasp.RLgrasp import RLGraspEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import NatureCNN
import torch
import torch.nn as nn

# 设置环境变量
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
os.environ['EGL_PLATFORM'] = 'device'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

def make_env(scene_id=0):
    def _init():
        env = RLGraspEnv(render_mode="rgb_array", grasp_mode="fixed_force", scene_range=range(88), scene_id=scene_id)  # 每个进程独立初始化
        return Monitor(env)
    return _init

class CustomCNN(NatureCNN):
    def __init__(self, *args, **kwargs):
        super(CustomCNN, self).__init__(*args, **kwargs)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(4, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(16, 64, kernel_size=3, stride=1),
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
        "n_steps": 8,                   # 每次更新的步数（PPO等）
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
            normalize_images=True,        # 是否归一化图像
        ),

        "verbose": 1,                   # 日志级别
        "tensorboard_log": "RLgrasp/logs/ppo_test",  # TensorBoard日志目录
        # "env_kwargs": dict(
        #     depth_size=84,                 # 深度图尺寸
        #     history_len=4,                 # 历史帧数
        #     reward_type="dense",           # 奖励类型
        # ),
        "train_envs": 24,                    # 并行环境数
        "eval_envs": 4,
        # "save_freq": 10_000,               # 保存频率
        "eval_freq": 24,                     # 验证频率
        # "seed": 42,                        # 随机种子
    }

    # # 创建多线程向量化环境
    # train_env = SubprocVecEnv([make_env(scene_id) for scene_id in range(config["train_envs"])])
    # eval_env = SubprocVecEnv([make_env(scene_id) for scene_id in range(config["eval_envs"])])
    # # env = DummyVecEnv([make_env()])
    # # eval_env = DummyVecEnv([make_env()])

    # 评估回调
    # eval_callback = EvalCallback(eval_env, best_model_save_path=config["tensorboard_log"], log_path=config["tensorboard_log"], eval_freq=config['eval_freq'])

    # # 创建模型
    # model = PPO("CnnPolicy", train_env, **{k: v for k, v in config.items() if k not in ["train_envs", "eval_envs", "eval_freq"]})
    # # model = PPO.load("RLgrasp/logs/ppo_test/best_model.zip", train_env)
    
    # # 训练
    # model.learn(total_timesteps=100000, callback=eval_callback)

    total_timesteps = 100000
    chunk = config["train_envs"] * config["eval_freq"]
    trained_steps = 0
    while trained_steps < total_timesteps:
        # 1. 创建新环境
        train_env = SubprocVecEnv([make_env(scene_id) for scene_id in range(config["train_envs"])])
        eval_env = SubprocVecEnv([make_env(scene_id) for scene_id in range(config["eval_envs"])])

        # 2. 只在第一次创建模型，后续直接加载
        if trained_steps == 0:
            model = PPO("CnnPolicy", train_env, **{k: v for k, v in config.items() if k not in ["train_envs", "eval_envs", "eval_freq"]})
        else:
            model.set_env(train_env)

        # 3. 评估回调
        eval_callback = EvalCallback(eval_env, best_model_save_path=config["tensorboard_log"], log_path=config["tensorboard_log"], eval_freq=config['eval_freq'])

        # 4. 训练一段
        try:
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=eval_callback)
            trained_steps += chunk
        except Exception as e:
            print("Chunk failed and reset.")
            continue

        # 5. 关闭环境，释放内存
        print(f"Trained {trained_steps}/{total_timesteps} steps. Envs are reset.")
        train_env.close()
        eval_env.close()
        del train_env
        del eval_env
        del eval_callback
        import gc, time
        gc.collect()
        time.sleep(1)

    # 保存模型
    model.save(f'{config["tensorboard_log"]}/final_model')

    # # 关闭环境
    # train_env.close()
    # eval_env.close()