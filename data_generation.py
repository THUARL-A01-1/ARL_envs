import cad.grasp_sampling
import cv2
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import mujoco    
from mujoco import viewer
from dexhand.dexhand import DexHandEnv
from scipy.spatial.transform import Rotation as R


def pre_grasp(env, point, normal, angle, depth):
    """Pre-grasp the object by moving the hand to the target position.
    Args: env (DexHandEnv): The DexHand environment. point, normal, depth: Target position of shape (3, 3, 1).
    Note: translation: qpos[0:3], rotation: qpos[3:6]
    """
    translation = point + normal * (depth + 0.13)  # 0.13 is the offset from the base mount to the center of the fingers
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
    env.step(np.array([0, 0, 0, 0, 0, 0, 10]))
    env.step(np.array([0, 0, 0, 0, 0, 0, -10]))
    env.step(np.array([0, 0, 0, 0, 0, 0, 10]))
    env.step(np.array([0, 0, 0.1, 0, 0, 0, 10]))
    # env.step(np.array([0, 0, 0, 0, 0, 0, 10]))

    # remove the gravity compensation
    body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
    env.mj_model.body_gravcomp[body_id] = 0.0
    env.step(np.array([0, 0, 0, 0, 0, 0, 10]))

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

        measurement.append({"P_field": P_field.tolist(), "N_field": N_field.tolist(), "N_field_finger": N_field_finger.tolist(), "Fn_field": Fn_field.tolist(), "Ft_field": Ft_field.tolist(), "Fv": Fv.tolist(), "F_mask": F_mask.tolist()})

    return measurement

def post_grasp(env):
    """Post-grasp the object by moving the hand, simulating the disturbance.
    Args: env (DexHandEnv): The DexHand environment.
    """
    for i in range(1):
        env.step(np.array([0, 0, 0.1, 0, 0, 0, 10]))
        env.step(np.array([0, 0, -0.1, 0, 0, 0, 10]))
        env.step(np.array([0.1, 0, 0, 0, 0, 0, 10]))
        env.step(np.array([-0.1, 0, 0, 0, 0, 0, 10]))
        env.step(np.array([0, 0.1, 0, 0, 0, 0, 10]))
        env.step(np.array([0, -0.1, 0, 0, 0, 0, 10]))

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
    success = bool(angle_rad < 0.3)  # threshold need to be modified

    if not contact_success(env):
        success = False
        
    floor_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(env.mj_data.ncon):  # 遍历接触对，判断物体是否与地面接触
        geom_id1, geom_id2 = env.mj_data.contact[i].geom1, env.mj_data.contact[i].geom2
        if geom_id1 == floor_id or geom_id2 == floor_id:
            success = False
            break
    
    return success

def calculate_FC_metric(measurement):
    num_fingers = len(measurement)
    metric = np.zeros(num_fingers)
    P_list, N_list, N_finger_list = [], [], []
    for i in range(num_fingers):
        F_mask = np.linalg.norm(measurement[i]["Fn_field"], axis=1) > 0.05
        P_field, N_field, N_field_finger = np.array(measurement[i]["P_field"]), np.array(measurement[i]["N_field"]), np.array(measurement[i]["N_field_finger"])
        P, N, N_finger = np.mean(P_field[F_mask], axis=0), np.mean(N_field[F_mask], axis=0), np.mean(N_field_finger[F_mask], axis=0)
        P_list.append(P)
        N_list.append(N)
        N_finger_list.append(N_finger)
        # print(f"Finger {i+1}: P: {P}, N: {N}, N_finger: {N_finger}")
    # the angle of N_list[0] and N_list[1]
    alpha = -np.dot(N_list[0], N_list[1]) / (np.linalg.norm(N_list[0]) * np.linalg.norm(N_list[1]))
    beta = np.dot(N_finger_list[0], np.array([0, 0, 1])) / (np.linalg.norm(N_finger_list[0])) + np.dot(N_finger_list[1], np.array([0, 0, 1])) / (np.linalg.norm(N_finger_list[1]))
    metric = np.array([alpha, beta])
    distance = np.linalg.norm(np.mean(np.array(P_list), axis=0) - np.array([0.07, 0.07, 0.04]))

    return metric, distance


def calculate_our_metric(measurement):
    num_fingers = len(measurement)
    metric = np.zeros(num_fingers)
    Fv = np.zeros(num_fingers)
    for i in range(num_fingers):
        
        # F_field = np.array(measurement[i]["Ft_field"]) + np.array(measurement[i]["Fn_field"])
        # F_field_corrected = np.zeros_like(F_field)
        # for j in range(400):
        #     f = F_field[j]
        #     if np.linalg.norm(f) > 0.02:
        #         dx, dy = j % 20 - 10, j // 20 - 10
        #         t = np.array([-dy, dx, 0]) / np.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        #         sin = -np.sqrt(dx ** 2 + dy ** 2 + 1e-6) / 35
        #         cos = np.sqrt(1 - sin ** 2)
        #         f_parallel = np.dot(f, t) * t
        #         f_vertical = f - f_parallel
        #         f_corrected = f_parallel + cos * f_vertical + sin * np.cross(t, f_vertical)
        #         F_field_corrected[j] = f_corrected
        # F_mask = np.linalg.norm(F_field_corrected, axis=1) > 0.1
        # ratio = np.linalg.norm(F_field_corrected[:, :2], axis=1)

        # F_mask = np.linalg.norm(measurement[i]["Fn_field"], axis=1) > 0.05
        # ratio = np.linalg.norm(measurement[i]["Ft_field"], axis=1) / np.linalg.norm(measurement[i]["Fn_field"], axis=1)
        
        # metric[i] = sum(ratio[F_mask]) / (sum(F_mask))

        metric[i] = np.sum(np.linalg.norm(measurement[i]["Ft_field"], axis=1)) / np.sum(np.linalg.norm(measurement[i]["Fn_field"], axis=1))

        Fv[i] = measurement[i]["Fv"]
        # print(f"Finger {i+1}: Metric: {sum(ratio[F_mask]) / sum(F_mask)}, Fv: {Fv[i]}")
    
    return metric, Fv
    
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

    for i in range(len(grasp_points)):
        _ = env.reset()
        # pre-grasp the object
        pre_grasp(env, grasp_points[i], grasp_normals[i], grasp_angles[i], grasp_depths[i])
        
        # grasp the object
        grasp(env)
        contact_result = contact_success(env)
        if contact_result == False:
            print(f"Grasp {i+1}/{len(grasp_points)}: Contact Failed, skipping...")
            continue
        measurement = measure(env)
        if measurement[0]["F_mask"].count(True) < 20 or measurement[1]["F_mask"].count(True) < 20:  # Check if the contact is sufficient
            print(f"Grasp {i+1}/{len(grasp_points)}: Insufficient contact, skipping...")
            continue
        our_metric, Fv = calculate_our_metric(measurement)
        FC_metric, distance = calculate_FC_metric(measurement)

        # post-grasp the object
        grasp_result = grasp_success(env)
        
        print(f"Grasp {i+1}/{len(grasp_points)}: Contact Success: {contact_result}, Grasp Success: {grasp_result}, Our Metric: {our_metric}, FC Metric: {FC_metric}, Distance: {distance}, Fv: {Fv}")

        # save the results
        result = {
            "contact_result": contact_result,
            "grasp_result": grasp_result,
            "measurement": measurement}
        with open(f"results/{OBJECT_ID}/grasp_results.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if contact_result == True and ((grasp_result == True and np.mean(our_metric) > 0.6) or (grasp_result == False and np.mean(our_metric) < 0.3)):  # Filter out the grasps that are not rational
            our_metric, Fv = calculate_our_metric(measurement)
            FC_metric, distance = calculate_FC_metric(measurement)
            env.render()

    
def preprocess_results(OBJECT_ID):
    measurement_list = []
    with open(f"results/{OBJECT_ID}/grasp_results.json", "r", encoding="utf-8") as f:
        print(f"The number of grasps: {len(f.readlines())}")
        f.seek(0)
        for line in f:
            result = json.loads(line)
            contact_result = result["contact_result"]
            grasp_result = result["grasp_result"]
            measurement = result["measurement"]
            if contact_result == True:
                our_metric, Fv = calculate_our_metric(measurement)
                FC_metric, distance = calculate_FC_metric(measurement)
                if np.sum(FC_metric) > 2 and np.mean(our_metric) > 0.8:  # Filter out the grasps that are not rational
                    our_metric, Fv = calculate_our_metric(measurement)
                    FC_metric, distance = calculate_FC_metric(measurement)

                measurement_list.append({
                    "grasp_result": grasp_result,
                    "our_metric": our_metric,
                    "FC_metric": FC_metric,
                    "distance": distance,
                    "Fv": Fv
                })
    # save the results to a npz file
    np.savez(f"results/{OBJECT_ID}/grasp_metrics.npz", 
             grasp_results=[result["grasp_result"] for result in measurement_list],
             our_metrics=[result["our_metric"] for result in measurement_list],
             FC_metrics=[result["FC_metric"] for result in measurement_list],
             distances=[result["distance"] for result in measurement_list],
             Fvs=[result["Fv"] for result in measurement_list])
    
def combine_results():
    grasp_results_all, our_metrics_all, FC_metrics_all, distances_all, Fvs_all  = [], [], [], [], []
    for i in range(1, 10):
        # if i == 18: 
        #     continue
        OBJECT_ID = f"{i:03d}"
        print(f"Processing object {OBJECT_ID}...")
        data = np.load((f"results/{OBJECT_ID}/grasp_metrics.npz"))
        grasp_results, our_metrics, FC_metrics, distances, Fvs= data['grasp_results'], data['our_metrics'], data['FC_metrics'], data['distances'], data['Fvs']
        grasp_results_all.extend(grasp_results)
        our_metrics_all.extend(our_metrics)
        FC_metrics_all.extend(FC_metrics)
        distances_all.extend(distances)
        Fvs_all.extend(np.abs(Fvs))  # Use absolute value of Fv
    # save the results to a npz file
    if not os.path.exists("results/all"):
        os.makedirs("results/all")
    np.savez("results/all/grasp_metrics.npz",
             grasp_results=grasp_results_all,
             our_metrics=our_metrics_all,
             FC_metrics=FC_metrics_all,
             distances=distances_all,
             Fvs=Fvs_all)


def validate_result(OBJECT_ID):
    data = np.load((f"results/{OBJECT_ID}/grasp_metrics.npz"))
    grasp_results, our_metrics, FC_metrics, distances, Fvs = data['grasp_results'], data['our_metrics'], data['FC_metrics'], data['distances'], data['Fvs']
    grasp_results = data['grasp_results']
    our_metrics = np.mean(data['our_metrics'], axis=1)  # Combine the metrics from both fingers
    our_metrics = np.nan_to_num(our_metrics, nan=10)  # Replace NaN with 100
    FC_metrics = np.sum(data['FC_metrics'], axis=1)  # Combine the metrics from both fingers
    FC_metrics = np.nan_to_num(FC_metrics, nan=10)  # Replace NaN with 100
    distances = data['distances']
    Fvs = np.abs(np.sum(data['Fvs'], axis=1))
    # our_metrics = our_metrics / (Fvs + 1e-6)  # Normalize the our metrics by Fv
    # FC_metrics = FC_metrics / 0.1 * (distances + 1e-6)  # Normalize the FC metrics by distance

    mask = (our_metrics < 1) & (FC_metrics < 3)  # Filter out the metrics that are too large
    FC_metrics, our_metrics, grasp_results, distances, Fvs = FC_metrics[mask], our_metrics[mask], grasp_results[mask], distances[mask], Fvs[mask]
    metrics = our_metrics  # Normalize the our metrics by Fv
    
    # 绘制散点图，横轴为our_metric，纵轴为FC_metric
    from scipy.stats import pearsonr
    corr, pval = pearsonr(metrics, FC_metrics)
    print(f"our_metric 与 metrics 的皮尔逊相关系数: {corr:.4f}, p值: {pval:.4e}")

    plt.scatter(metrics[grasp_results == True], FC_metrics[grasp_results == True], alpha=0.7, label='Grasp Success', color='blue', s=1)
    plt.scatter(metrics[grasp_results == False], FC_metrics[grasp_results == False], alpha=0.7, label='Grasp Failure', color='red', s=1)
    plt.xlabel('Our Metric')
    plt.ylabel('FC Metric')
    plt.title('Our Metric vs FC Metric')
    plt.legend()
    plt.show()

    # 绘制两个直方图，分别是grasp成功和失败的our_metric分布
    plt.hist(metrics[grasp_results == True], bins=200, alpha=0.5, label='Grasp Success', color='blue')
    plt.hist(metrics[grasp_results == False], bins=200, alpha=0.5, label='Grasp Failure', color='red')
    plt.xlabel('Our Metric')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # 计算AUROC
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(grasp_results, -metrics)
    print(f"AUROC: {auroc:.4f}")

    # 假设检验
    from scipy.stats import ks_2samp

    # 假设 data1, data2 是两个一维数组
    stat, p_value = ks_2samp(metrics[grasp_results == True], metrics[grasp_results == False])
    print(f"KS检验统计量: {stat}")
    print(f"KS检验 p值: {p_value}")


if __name__ == '__main__':

    import shutil
    base_dir = "E:/2 - 3_Technical_material/Simulator/ARL_envs/cad/assets"
    for i in range(4, 5):
        OBJECT_ID = f"{i:03d}"
        print(f"Processing object {OBJECT_ID}...")

        # Simulate the grasping process
        src = os.path.join(base_dir, OBJECT_ID, "downsampled_mesh.obj")
        dst = os.path.join(base_dir, "downsampled_mesh.obj")
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            print(f"Source not found: {src}")
        simulate(OBJECT_ID=OBJECT_ID, num_samples=20)

        # Preprocess the results after simulation
        preprocess_results(OBJECT_ID=OBJECT_ID)

        # Validate the results
        validate_result(OBJECT_ID=OBJECT_ID)
        
    
    # combine_results()
    # validate_result(OBJECT_ID="all") 