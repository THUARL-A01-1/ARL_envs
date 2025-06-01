import cad.grasp_sampling
import cv2
import json
import open3d as o3d
import os
import matplotlib.pyplot as plt
import numpy as np
import function.metrics as metrics
import mujoco    
from mujoco import viewer
from dexhand.dexhand import DexHandEnv
from scipy.spatial.transform import Rotation as R


def pre_grasp(env, point, normal, angle, depth):
    """Pre-grasp the object by moving the hand to the target position.
    Args: env (DexHandEnv): The DexHand environment. point, normal, depth: Target position of shape (3, 3, 1).
    Note: translation: qpos[0:3], rotation: qpos[3:6]
    """
    translation = point + normal * (depth + 0.15)  # 0.13 is the offset from the base mount to the center of the fingers
    R_to_normal, _ = R.align_vectors([normal], [[0, 0, 1]])
    R_about_normal = R.from_rotvec(angle * normal)
    rot = R_about_normal * R_to_normal
    rotation = rot.as_euler('XYZ', degrees=False)
    env.mj_data.qpos[0:3] = translation
    env.mj_data.qpos[3:6] = rotation
    env.step(np.array([0, 0, 0, 0, 0, 0, 0]))

def grasp(env):
    """Grasp the object by applying a force to the hand, and then gravity.
    Args: env (DexHandEnv): The DexHand environment.
    """
    # apply grasping force
    env.step(np.array([0, 0, 0, 0, 0, 0, 3]))
    env.step(np.array([0, 0, 0.1, 0, 0, 0, 3]))

    # remove the gravity compensation
    body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
    env.mj_model.body_gravcomp[body_id] = 0.0
    env.step(np.array([0, 0, 0, 0, 0, 0, 3]))

def contact_success(env):
    """Check if the object is in contact with the hand.
    Args: env (DexHandEnv): The DexHand environment.
    Returns: bool: True if the object is in contact with the hand, False otherwise.
    """
    finger_geom_idx_list = [[mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"left_pad_collisions_{i}") for i in range(400)], [mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"right_pad_collisions_{i}") for i in range(400)]]
    object_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "object")
    success = False
    for i in range(env.mj_data.ncon):  # 遍历接触对，判断物体是否与手指接触
        geom_id1, geom_id2 = env.mj_data.contact[i].geom1, env.mj_data.contact[i].geom2
        if (geom_id1 == object_id and any(geom_id2 in sublist for sublist in finger_geom_idx_list)) or (geom_id2 == object_id and any(geom_id1 in sublist for sublist in finger_geom_idx_list)):
            success = True
            break

    return success

def measure(env):
    """
    Measure the contact normals and forces of the grasps.
    Args: env (DexHandEnv): The DexHand environment.
    Consts: rotation_left/right: rotation matrix relative to the hand
            geom_idx_left/right: geom indexes of the mirco units
    Returns: Dict{List[np.adarray]}: positions, normals, and the forces of the fingers
    """
    # 常量，按照手指顺序
    num_fingers = 2  # 黑色driver为左，白色driver为右
    finger_rotation_list = [np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])]
    finger_geom_idx_list = [[mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"left_pad_collisions_{i}") for i in range(400)], [mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"right_pad_collisions_{i}") for i in range(400)]]

    rotation_hand = env.mj_data.geom_xmat[4].reshape(3, 3)
    object_pos = env.mj_data.qpos[8:11].copy()
    measurement = []
    for i in range(num_fingers):
        rotation, geom_idx = finger_rotation_list[i], finger_geom_idx_list[i]
        # Step 1: 初始化PNF矩阵（F矩阵为手指坐标系）
        P_field = env.mj_data.geom_xpos[geom_idx]
        F_field = env.mj_data.sensordata[1200 * i:1200 * (i + 1)].reshape(3, -1).T
        F_field = np.roll(F_field, -1, axis=1)  # [z, x, y] -> [x, y, z]
        F_field[:, 0] = -F_field[:, 0]  # x-axis is flipped
        N_field = F_field / (0.001 + np.linalg.norm(F_field, axis=1)[:, np.newaxis])
        # Step 2: 使用碰撞对为PN矩阵赋值（PN矩阵为世界坐标系）
        for i in range(env.mj_data.ncon):
            geom_id = env.mj_data.contact[i].geom1  # 获取第i个接触几何体的索引geom_id
            if geom_id in geom_idx:  # 若geom_id属于该手指，则保存该几何体数据
                geom_id = geom_idx.index(geom_id)
                geom_id = 20 * (geom_id // 20) + (19 - geom_id % 20)
                P_field[geom_id] = env.mj_data.contact[i].pos[:3]
                N_field[geom_id] = env.mj_data.contact[i].frame[:3]
        # Step 3: 使用旋转矩阵计算手指坐标系下的N矩阵, 用于接触力分解
        N_field_finger = N_field @ rotation_hand @ rotation
        Fn_field = np.sum(N_field_finger * F_field, axis=1)[:, np.newaxis] * N_field_finger
        Ft_field = F_field - Fn_field

        # Step 4: 额外计算用于抵抗重力的竖直力
        F_field_world = F_field @ rotation.T @ rotation_hand.T   # 将F矩阵转换到世界坐标系
        Fv = np.sum(F_field_world[:, 2])
        F_mask = np.linalg.norm(F_field, axis=1) > 0.05

        measurement.append({"P_field": P_field.tolist(), "N_field": N_field.tolist(), "N_field_finger": N_field_finger.tolist(), "Fn_field": Fn_field.tolist(), "Ft_field": Ft_field.tolist(), "Fv": Fv.tolist(), "F_mask": F_mask.tolist(), "rotation_hand": rotation_hand.tolist(), "object_pos": object_pos.tolist()})

    return measurement

def post_grasp(env):
    """Post-grasp the object by moving the hand, simulating the disturbance.
    Args: env (DexHandEnv): The DexHand environment.
    """
    for i in range(1):
        env.step(np.array([0, 0, 0.1, 0, 0, 0, 3]))
        # env.step(np.array([0, 0, 0.1, 0, 0, 0, 3]))
        # env.step(np.array([0.1, 0, 0, 0, 0, 0, 10]))
        # env.step(np.array([-0.1, 0, 0, 0, 0, 0, 10]))
        # env.step(np.array([0, 0.1, 0, 0, 0, 0, 10]))
        # env.step(np.array([0, -0.1, 0, 0, 0, 0, 10]))

def grasp_success(env):
    """
    calculate the empirical metirc under the action disturbance.
    Args: env (DexHandEnv): The DexHand environment.
    Returns: whether the object contacts the floor.
    """
    success = True
    object_quat0 = env.mj_data.qpos[11:].copy()  # 记录抓取前的物体姿态
    post_grasp(env)  # post-grasp the object to simulate the disturbance
    object_quat1 = env.mj_data.qpos[11:].copy()
    rot = R.from_quat(object_quat1[[1,2,3,0]]) * R.from_quat(object_quat0[[1,2,3,0]]).inv()  # 计算物体的旋转矩阵
    angle_rad = rot.magnitude()  # 旋转弧度
    success = bool(angle_rad < 1.57)  # threshold need to be modified

    if not contact_success(env):
        success = False
        
    floor_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(env.mj_data.ncon):  # 遍历接触对，判断物体是否与地面接触
        geom_id1, geom_id2 = env.mj_data.contact[i].geom1, env.mj_data.contact[i].geom2
        if geom_id1 == floor_id or geom_id2 == floor_id:
            success = False
            break
    
    return success

    
def simulate(OBJECT_ID, num_samples=500):
    # initialize the environment
    env = DexHandEnv()
    _ = env.reset()
    if not os.path.exists(f"results/{OBJECT_ID}"):
        os.makedirs(f"results/{OBJECT_ID}")

    # sample grasps from the CAD model
    try:
        grasp_points, grasp_normals, grasp_angles, grasp_depths = cad.grasp_sampling.main(num_samples=num_samples, OBJECT_ID=OBJECT_ID)
    except Exception as e:
        print(f"未生成无碰撞抓取, Error sampling grasps: {e}")
        return
    file_path = f"cad/assets/{OBJECT_ID}/downsampled.ply"  # Replace with your point cloud file path
    point_cloud = o3d.io.read_point_cloud(file_path)
    centroid = np.mean(np.asarray(point_cloud.points), axis=0)
    print(f"The initial centroid of the object {OBJECT_ID}: {centroid}")

    for i in range(len(grasp_points)):
        _ = env.reset()
        # pre-grasp the object
        pre_grasp(env, grasp_points[i], grasp_normals[i], grasp_angles[i], grasp_depths[i])
        
        # grasp the object
        grasp(env)
        contact_result = contact_success(env)
        if contact_result == False:
            print(f"Grasp {i+1}/{len(grasp_points)}: Contact Failed, skipping...")
            # env.render()
            continue
        measurement = measure(env)
        
        if measurement[0]["F_mask"].count(True) < 10 or measurement[1]["F_mask"].count(True) < 10:  # Check if the contact is sufficient
            print(f"Grasp {i+1}/{len(grasp_points)}: Insufficient contact, skipping...")
            continue

        centroid += np.array(measurement[0]["object_pos"])
        our_metric, Fv = metrics.calculate_our_metric(measurement)
        antipodal_metric, distance = metrics.calculate_antipodal_metric(measurement)
        closure_metric = metrics.calculate_closure_metric(measurement, centroid)

        # post-grasp the object
        grasp_result = grasp_success(env)
        
        print(f"Grasp {i+1}/{len(grasp_points)}: Contact Success: {contact_result}, Grasp Success: {grasp_result}, Our Metric: {our_metric}, antipodal Metric: {antipodal_metric}, closure_metric, {closure_metric}, Distance: {distance}, Fv: {Fv}")

        # save the results
        result = {
            "contact_result": contact_result,
            "grasp_result": grasp_result,
            "measurement": measurement}
        with open(f"results/{OBJECT_ID}/grasp_results.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if contact_result == True and ((grasp_result == True and np.mean(our_metric) > 0.6) or (grasp_result == False and np.mean(our_metric) < 0.4)):  # Filter out the grasps that are not rational
            our_metric, Fv = metrics.calculate_our_metric(measurement)
            antipodal_metric, distance = metrics.calculate_antipodal_metric(measurement)
            # env.render()

    
def preprocess_results(OBJECT_ID):
    measurement_list = []
    # 读取对应的mesh
    file_path = f"cad/assets/{OBJECT_ID}/downsampled.ply"  # Replace with your point cloud file path
    point_cloud = o3d.io.read_point_cloud(file_path)
    centroid = np.mean(np.asarray(point_cloud.points), axis=0)
    print(f"The initial centroid of the object {OBJECT_ID}: {centroid}")

    with open(f"results/{OBJECT_ID}/grasp_results.json", "r", encoding="utf-8") as f:
        print(f"The number of grasps: {len(f.readlines())}")
        f.seek(0)
        for line in f:
            result = json.loads(line)
            contact_result = result["contact_result"]
            grasp_result = result["grasp_result"]
            measurement = result["measurement"]
            if contact_result == True:
                centroid += np.array(measurement[0]["object_pos"])  # Update the centroid with the object position
                our_metric, Fv = metrics.calculate_our_metric(measurement)
                antipodal_metric, distance = metrics.calculate_antipodal_metric(measurement, centroid)
                closure_metric = metrics.calculate_closure_metric(measurement, centroid)
                # if np.sum(antipodal_metric) > 2 and np.mean(our_metric) > 0.8:  # Filter out the grasps that are not rational
                #     our_metric, Fv = metrics.calculate_our_metric(measurement)
                #     antipodal_metric, distance = metrics.calculate_closure_metric(measurement, centroid)

                measurement_list.append({
                    "grasp_result": grasp_result,
                    "our_metric": our_metric,
                    "antipodal_metric": antipodal_metric,
                    "closure_metric": closure_metric,
                    "distance": distance,
                    "Fv": Fv
                })
    # save the results to a npz file
    np.savez(f"results/{OBJECT_ID}/grasp_metrics.npz", 
             grasp_results=[result["grasp_result"] for result in measurement_list],
             our_metrics=[result["our_metric"] for result in measurement_list],
             antipodal_metrics=[result["antipodal_metric"] for result in measurement_list],
             closure_metrics=[result["closure_metric"] for result in measurement_list],
             distances=[result["distance"] for result in measurement_list],
             Fvs=[result["Fv"] for result in measurement_list])
    
def combine_results():
    grasp_results_all, our_metrics_all, antipodal_metrics_all, closure_metrics_all, distances_all, Fvs_all  = [], [], [], [], [], []
    for i in range(89):
        # if i == 9 or i == 45: 
        #     continue
        OBJECT_ID = f"{i:03d}"
        print(f"Processing object {OBJECT_ID}...")
        data = np.load((f"results/{OBJECT_ID}/grasp_metrics.npz"))
        grasp_results, our_metrics, antipodal_metrics, closure_metrics, distances, Fvs= data['grasp_results'], data['our_metrics'], data['antipodal_metrics'], data['closure_metrics'], data['distances'], data['Fvs']
        grasp_results_all.extend(grasp_results)
        our_metrics_all.extend(our_metrics)
        antipodal_metrics_all.extend(antipodal_metrics)
        closure_metrics_all.extend(closure_metrics)
        distances_all.extend(distances)
        Fvs_all.extend(np.abs(Fvs))  # Use absolute value of Fv
    # save the results to a npz file
    if not os.path.exists("results/all"):
        os.makedirs("results/all")
    np.savez("results/all/grasp_metrics.npz",
             grasp_results=grasp_results_all,
             our_metrics=our_metrics_all,
             antipodal_metrics=antipodal_metrics_all,
             closure_metrics=closure_metrics_all,
             distances=distances_all,
             Fvs=Fvs_all)


def validate_result(OBJECT_ID):
    data = np.load((f"results/{OBJECT_ID}/grasp_metrics.npz"))
    grasp_results, our_metrics, antipodal_metrics, distances, Fvs = data['grasp_results'], data['our_metrics'], data['antipodal_metrics'], data['distances'], data['Fvs']
    grasp_results = data['grasp_results']
    our_metrics = np.mean(data['our_metrics'], axis=1)  # Combine the metrics from both fingers
    our_metrics = np.nan_to_num(our_metrics, nan=10)  # Replace NaN with 100
    antipodal_metrics = np.sum(data['antipodal_metrics'], axis=1)  # Combine the metrics from both fingers
    antipodal_metrics = np.nan_to_num(antipodal_metrics, nan=10)  # Replace NaN with 100
    closure_metrics = data['closure_metrics']  # Combine the metrics from both fingers
    closure_metrics = np.nan_to_num(closure_metrics, nan=10)  # Replace NaN with 100
    distances = data['distances']
    Fvs = np.abs(np.sum(data['Fvs'], axis=1))
    # our_metrics = our_metrics * (10*distances + 1e-6)  # Normalize the our metrics by Fv
    # antipodal_metrics = antipodal_metrics * (10 * distances + 1e-6)  # Normalize the antipodal metrics by distance

    mask = (our_metrics < 0.999) & (closure_metrics > 0) & (closure_metrics < 3.0)# & (distances > 0.03)  # Filter out the metrics that are too large
    our_metrics, antipodal_metrics, closure_metrics, grasp_results, distances, Fvs = our_metrics[mask], antipodal_metrics[mask], closure_metrics[mask], grasp_results[mask], distances[mask], Fvs[mask]
    metrics = our_metrics  # Normalize the our metrics by Fv
    
    # 绘制散点图，横轴为our_metric，纵轴为antipodal_metric
    from scipy.stats import pearsonr
    corr, pval = pearsonr(metrics, closure_metrics)
    print(f"our_metric 与 metrics 的皮尔逊相关系数: {corr:.4f}, p值: {pval:.4e}")

    plt.scatter(metrics[grasp_results == True], closure_metrics[grasp_results == True], alpha=0.7, label='Grasp Success', color='blue', s=0.1)
    plt.scatter(metrics[grasp_results == False], closure_metrics[grasp_results == False], alpha=0.7, label='Grasp Failure', color='red', s=0.1)
    plt.xlabel('Our Metric')
    plt.ylabel('antipodal Metric')
    plt.title('Our Metric vs antipodal Metric')
    plt.legend()
    plt.show()

    # 绘制两个直方图，分别是grasp成功和失败的our_metric分布
    plt.hist(metrics[grasp_results == True], bins=100, alpha=0.5, label='Grasp Success', color='blue', density=False, orientation='vertical')
    plt.hist(metrics[grasp_results == False], bins=100, alpha=0.5, label='Grasp Failure', color='red', density=False, orientation='vertical')
    plt.xlabel('Our Metric')
    plt.ylabel('Density')
    plt.legend()
    # plt.gca().invert_xaxis()
    # plt.gca().xaxis.tick_top()
    # plt.gca().xaxis.set_label_position("top")
    plt.show()

    # 计算AUROC
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(grasp_results, -metrics)
    print(f"1. AUROC: {auroc:.4f}")
    # # 假设检验
    # from scipy.stats import ks_2samp
    # stat, p_value = ks_2samp(metrics[grasp_results == True], metrics[grasp_results == False])
    # print(f"2. KS检验统计量: {stat}, p值: {p_value}")
    # from scipy.stats import wasserstein_distance
    # d = wasserstein_distance(metrics[grasp_results == True], metrics[grasp_results == False])
    # print(f"3. Wasserstein 距离: {d}")
    # from scipy.stats import mannwhitneyu
    # stat, p = mannwhitneyu(metrics[grasp_results == True], metrics[grasp_results == False], alternative='two-sided')
    # print(f"4. Mann-Whitney U 检验统计量: {stat}, p值: {p}")
    # cohen_d = (np.mean(metrics[grasp_results == True]) - np.mean(metrics[grasp_results == False])) / np.sqrt((np.std(metrics[grasp_results == True], ddof=1) ** 2 + np.std(metrics[grasp_results == False], ddof=1) ** 2) / 2)
    # print(f"5. Cohen's d: {cohen_d}")
    # from scipy.spatial.distance import jensenshannon
    # # 先对数据做直方图归一化
    # hist1, bins = np.histogram(metrics[grasp_results == True], bins=100, density=True)
    # hist2, _ = np.histogram(metrics[grasp_results == False], bins=bins, density=True)
    # jsd = jensenshannon(hist1, hist2)
    # print(f"6. Jensen-Shannon 距离: {jsd}")
    # from scipy.stats import ttest_ind
    # stat, p = ttest_ind(metrics[grasp_results == True], metrics[grasp_results == False])
    # print(f"7. t检验统计量: {stat}, p值: {p}")


if __name__ == '__main__':

    import shutil
    base_dir = r"E:\2 - 3_Technical_material\Simulator\ARL_envs\cad\assets"
    for i in range(89):
        OBJECT_ID = f"{i:03d}"
        # print(f"Processing object {OBJECT_ID}...")

        # Simulate the grasping process
        src = os.path.join(base_dir, OBJECT_ID, "downsampled_mesh.obj")
        dst = os.path.join(base_dir, "downsampled_mesh.obj")
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            print(f"Source not found: {src}")
        simulate(OBJECT_ID=OBJECT_ID, num_samples=50)

        # Preprocess the results after simulation
        preprocess_results(OBJECT_ID=OBJECT_ID)

        # # Validate the results
        # validate_result(OBJECT_ID=OBJECT_ID)
        
    
    # combine_results()
    # validate_result(OBJECT_ID="all") 