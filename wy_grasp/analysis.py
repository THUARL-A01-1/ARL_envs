import matplotlib.pyplot as plt
import numpy as np
import os


# ROOT_DIR = "E:/2 - 3_Technical_material/Simulator/ARL_envs"
ROOT_DIR = "/home/ad102/AutoRobotLab/projects/Simulation/ARL_envs"

def load_results(OBJECT_ID):
    """
    Analyze the grasp results for a given object ID by loading the metrics and plotting the results.
    Args:
        OBJECT_ID (str): The ID of the object to analyze.
    1. Load the grasp metrics from the npz file.
    2. Filter the metrics based on certain conditions.
    3. Plot the relationship between our metric and closure metric.
    4. Calculate and print the correlation coefficient between our metric and closure metric.
        """
    data = np.load(os.path.join(ROOT_DIR, f"results/{OBJECT_ID}/grasp_metrics.npz"))
    our_metrics = np.mean(data['our_metrics'], axis=1)  # Combine the metrics from both fingers
    our_metrics = np.nan_to_num(our_metrics, nan=1)  # Replace NaN with 100
    antipodal_metrics = np.sum(data['antipodal_metrics'], axis=1)  # Combine the metrics from both fingers
    antipodal_metrics = np.nan_to_num(antipodal_metrics, nan=1)  # Replace NaN with 100
    closure_metrics = data['closure_metrics']  # Combine the metrics from both fingers
    closure_metrics = np.nan_to_num(closure_metrics, nan=1)  # Replace NaN with 100
    Fvs = np.abs(np.sum(data['Fvs'], axis=1))
    object_ids, grasp_ids, grasp_results, friction_coefs, distances = data['object_ids'], data['grasp_ids'], data['grasp_results'], data['friction_coefs'], data['distances']
    
    # our_metrics = our_metrics / (np.power(Fvs, 0.15) + 1e-6)  # Normalize the our metrics by Fv
    # antipodal_metrics = antipodal_metrics * (10 * np.power(distances, 0.15) + 1e-6)  # Normalize the antipodal metrics by distance

    mask = (our_metrics > 0) & (our_metrics < 1.999) & (closure_metrics > 0) & (closure_metrics < 1.999) & (friction_coefs >= 0.5) & (grasp_results==True)# & (distances < 0.005)  # Filter out the metrics that are too large
    object_ids, grasp_ids, our_metrics, antipodal_metrics, closure_metrics, grasp_results, friction_coefs, distances, Fvs = object_ids[mask], grasp_ids[mask], our_metrics[mask], antipodal_metrics[mask], closure_metrics[mask], grasp_results[mask], friction_coefs[mask], distances[mask], Fvs[mask]

    print(f"Number of masked grasps: {np.sum(mask)}")
    if np.sum(mask) < 2:
        print("Masked grasps is not enough.")
        return
    
    return object_ids, grasp_ids, grasp_results, our_metrics, closure_metrics, antipodal_metrics, friction_coefs, distances, Fvs
    
def draw_histogram(grasp_results, metrics, closure_metrics):
    # 绘制散点图，横轴为our_metric，纵轴为antipodal_metric
    plt.scatter(metrics[grasp_results == True], closure_metrics[grasp_results == True], alpha=0.7, label=f"Grasp Success: {np.sum(grasp_results == True)}", color='blue', s=1)
    plt.scatter(metrics[grasp_results == False], closure_metrics[grasp_results == False], alpha=0.7, label=f"Grasp Failure: {np.sum(grasp_results == False)}", color='red', s=1)
    plt.xlabel('Our Metric')
    plt.ylabel('Closure Metric')
    plt.title('Our Metric vs antipodal Metric')
    plt.legend()
    plt.show()

    # 绘制两个直方图，分别是grasp成功和失败的our_metric分布
    plt.hist(metrics[grasp_results == True], bins=100, alpha=0.5, label='Grasp Success', color='blue', density=True, orientation='vertical')
    plt.hist(metrics[grasp_results == False], bins=100, alpha=0.5, label='Grasp Failure', color='red', density=True, orientation='vertical')
    plt.xlabel('Our Metric')
    plt.ylabel('Density')
    plt.legend()
    # plt.gca().invert_xaxis()
    # plt.gca().xaxis.tick_top()
    # plt.gca().xaxis.set_label_position("top")
    plt.show()

def analyze_classification_results(grasp_results, metrics):
    # 计算AUROC
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(grasp_results, -metrics)
    print(f"1. AUROC: {auroc:.4f}")
    # 假设检验
    from scipy.stats import ks_2samp
    stat, p_value = ks_2samp(metrics[grasp_results == True], metrics[grasp_results == False])
    print(f"2. KS检验统计量: {stat}, p值: {p_value}")
    from scipy.stats import wasserstein_distance
    d = wasserstein_distance(metrics[grasp_results == True], metrics[grasp_results == False])
    print(f"3. Wasserstein 距离: {d}")
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(metrics[grasp_results == True], metrics[grasp_results == False], alternative='two-sided')
    print(f"4. Mann-Whitney U 检验统计量: {stat}, p值: {p}")
    cohen_d = (np.mean(metrics[grasp_results == True]) - np.mean(metrics[grasp_results == False])) / np.sqrt((np.std(metrics[grasp_results == True], ddof=1) ** 2 + np.std(metrics[grasp_results == False], ddof=1) ** 2) / 2)
    print(f"5. Cohen's d: {cohen_d}")
    from scipy.spatial.distance import jensenshannon
    # 先对数据做直方图归一化
    hist1, bins = np.histogram(metrics[grasp_results == True], bins=100, density=True)
    hist2, _ = np.histogram(metrics[grasp_results == False], bins=bins, density=True)
    jsd = jensenshannon(hist1, hist2)
    print(f"6. Jensen-Shannon 距离: {jsd}")
    from scipy.stats import ttest_ind
    stat, p = ttest_ind(metrics[grasp_results == True], metrics[grasp_results == False])
    print(f"7. t检验统计量: {stat}, p值: {p}")

def analyze_correlation(metrics, closure_metrics):
    # 计算相关系数
    from scipy.stats import pearsonr
    corr, pval = pearsonr(metrics, closure_metrics)
    print(f"1. Our Metric 与 Closure Metric 的皮尔逊相关系数: {corr:.4f}, p值: {pval:.4e}")
    
    from scipy.stats import spearmanr
    corr, pval = spearmanr(metrics, closure_metrics)
    print(f"2. Our Metric 与 Closure Metric 的斯皮尔曼相关系数: {corr:.4f}, p值: {pval:.4e}")
    
    from scipy.stats import kendalltau
    corr, pval = kendalltau(metrics, closure_metrics)
    print(f"3. Our Metric 与 Closure Metric 的肯德尔相关系数: {corr:.4f}, p值: {pval:.4e}")

    # 将metrics取值0-1分为51个区间，每个区间计算closure_metrics的平均值和标准差
    means, stds = [], []
    bins = np.linspace(0, 1, 51)
    for i in range(len(bins) - 1):
        mask = (metrics >= bins[i]) & (metrics < bins[i + 1])
        if np.sum(mask) > 0:
            means.append(np.mean(closure_metrics[mask]))
            stds.append(np.std(closure_metrics[mask]))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    means, stds = np.array(means), np.array(stds)
    plt.errorbar(bins[:-1] + 0.005, means, yerr=stds, fmt='o', color="b", capsize=2, label='Closure Metric Mean ± Std')
    plt.xlabel('Our Metric')
    plt.ylabel('Distance Mean ± Std')
    plt.title('Distance Mean ± Std vs Our Metric')
    plt.legend()
    plt.show()

def analyze_correlation_friction(object_ids, grasp_ids, metrics):
    """
    Analyze the same grasp with five different friction_coefs, with the grasp_result==True
    """
    from collections import Counter
    # 统计 (object_id, grasp_id) 对出现的次数
    pairs = list(zip(object_ids, grasp_ids))
    counter = Counter(pairs)
    target_pairs = [pair for pair, count in counter.items() if count == 5]

    delta_metrics = []
    for target_pair in target_pairs:
        idx = [i for i, pair in enumerate(pairs) if pair == target_pair]
        delta_metric = metrics[idx]# - metrics[idx[0]]  # reduce the metric with mu=0.5
        delta_metrics.append(delta_metric)
    delta_metrics = np.array(delta_metrics)

    means, stds = np.mean(delta_metrics, axis=0), np.std(delta_metrics, axis=0)
    bins = np.linspace(0.5, 1.75, 6)
    plt.errorbar(bins[:-1] + 0.005, means, yerr=stds, fmt='o', color="b", capsize=2, label='Closure Metric Mean ± Std')
    plt.xlabel('Our Metric')
    plt.ylabel('Distance Mean ± Std')
    plt.title('Distance Mean ± Std vs Our Metric')
    plt.legend()
    plt.show()


def analyze_results(OBJECT_ID="all"):
    """
    Analyze the grasp results for a given object ID.
    Args:
        OBJECT_ID (str): The ID of the object to analyze. If "all", analyze all objects.
    """
    # load grasp results
    object_ids, grasp_ids, grasp_results, our_metrics, closure_metrics, antipodal_metrics, friction_coefs, distances, Fvs = load_results(OBJECT_ID)
    
    # Draw histogram
    draw_histogram(grasp_results, our_metrics, closure_metrics)
    
    # Analyze classification
    # analyze_classification_results(grasp_results, our_metrics)
    
    # Analyze correlation
    # analyze_correlation(antipodal_metrics, our_metrics)
    analyze_correlation_friction(object_ids, grasp_ids, our_metrics)


if __name__ == "__main__":
    analyze_results(OBJECT_ID="all")