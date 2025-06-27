# 根据logs/ppo/evalulations.npz绘制训练曲线
import numpy as np
import matplotlib.pyplot as plt

def plot_training_curve(log_file):
    data = np.load(log_file)
    timesteps = data['timesteps']
    results = data['results']
    ep_lengths = data['ep_lengths']

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    mean_results = np.mean(results, axis=1)
    std_results = np.std(results, axis=1)
    plt.plot(timesteps, mean_results, label='Mean Episode Reward', color='blue')
    plt.fill_between(timesteps, mean_results - std_results, mean_results + std_results, color='blue', alpha=0.2)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Training Curve - Episode Reward')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    mean_ep_lengths = np.mean(ep_lengths, axis=1)
    std_ep_lengths = np.std(ep_lengths, axis=1)
    plt.plot(timesteps, mean_ep_lengths, label='Mean Episode Length', color='orange')
    plt.fill_between(timesteps, mean_ep_lengths - std_ep_lengths, mean_ep_lengths + std_ep_lengths, color='orange', alpha=0.2)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Length')
    plt.title('Training Curve - Episode Length')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_file.replace('.npz', '.png'))
    plt.show()

if __name__ == "__main__":
    log_file = "RLgrasp/logs/ppo_test/evaluations.npz"
    plot_training_curve(log_file)
    print(f"Training curve saved as {log_file.replace('.npz', '.png')}")
