import matplotlib.pyplot as plt
import numpy as np

def calculate_friction_cone(N_vec, friction_coef=1, num_samples=8):
    """
    Sample the friction cone vectors based on the normal vector and friction coefficient.
    
    Args:
        N_vec (3,): The normal vector of the contact point.
        friction_coef (float): The friction coefficient.
        num_samples (int): The number of vectors for the approximate friction cone.
    
    Returns:
        np.ndarray: An array of shape (num_samples, 3) containing the sampled friction cone vectors.
    """
    # 计算垂直于N_vec的平面上的两个基向量 
    onehot = np.zeros(3)  # 定义一个独热向量，独热向量的非零元素是N_vec的最小值对应的索引
    onehot[np.argmin(np.abs(N_vec))] = 1.0
    basis_1 = np.cross(N_vec, onehot)  # (3,)
    basis_1 /= np.linalg.norm(basis_1)  # 归一化
    basis_2 = np.cross(N_vec, basis_1)  # (3,) 计算另一个基向量，使其与N_vec和basis_1正交
    basis_2 /= np.linalg.norm(basis_2)  # 归一化
    
    # 计算摩擦圆锥的向量
    thetas = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)  # (num_samples,)
    friction_cone = np.array([np.cos(thetas) * basis_1 * friction_coef + np.sin(thetas) * basis_2 * friction_coef + N_vec for thetas in thetas])  # (num_samples, 3) 

    return friction_cone


def calculate_G_matrix(P_field, N_field, centroid, friction_coef=1):
    """
    Calculate the G matrix of a single finger: w = G @ a
    Note: w (6, 1) is the wrench to the object center, a(N * 8, 1) is a series of vectors representing the friction cone of every contact point.
    Args:
        P_field (N, 3): The position field of the contact points.
        N_field (N, 3): The normal field of the contact points.
        centroid (3, 1): The centroid of the object.
    Returns:
        G (N * 8, 6): The G matrix of the finger.
    """
    num_points = P_field.shape[0]
    G = np.zeros((num_points * 8, 6))
    friction_cones = np.zeros((num_points * 8, 3))  # (N * 8, 3): The friction cone vectors for all contact points
    
    for i in range(num_points):
        P_vec = P_field[i] - centroid
        N_vec = N_field[i]
        friction_cone = calculate_friction_cone(N_vec, friction_coef=friction_coef, num_samples=8)  # (8, 3)
        W = np.array([[1, 0, 0, 0, P_vec[2], -P_vec[1]],
                      [0, 1, 0, -P_vec[2], 0, P_vec[0]],
                      [0, 0, 1, P_vec[1], -P_vec[0], 0]])  # (3, 6): The transition from contact force to the object wrench
        friction_cones[i * 8: (i + 1) * 8, :] = friction_cone  # (8, 3): The friction cone vectors for the current contact point
        G[i * 8: (i + 1) * 8, :] = friction_cone @ W  # (8, 6): The transition from the friction cone to the object wrench
    
    return G.copy(), friction_cones.copy()


def calculate_closure_metric(measurement, centroid=np.array([0, 0, 0]), friction=1, draw=False):
    """
    Calculate the closure metric of a grasp based on the contact points and their normals.
    Args:
        measurement (list): A list of dictionaries, each containing the contact points and normals for each finger.
        centroid (np.ndarray): The centroid of the object.
        friction (float): The friction coefficient for the contact points.
        draw (bool): Whether to draw the 3D plot of the contact points and normals for debugging.
    Returns:
        tuple: A tuple containing the closure metric and the distance from the centroid to the mean contact point.
    Note: The closure metric is calculated by the following optimization problem:
        max f, s.t. G_eq @ a =f, G_ub @ a = 0, a >= 0, sum(a) = 1
        This solution's meaning is the maximum force that can be applied to the object.
    """
    # Calculate the G_total matrix for all fingers
    num_fingers = len(measurement)
    G_total, G_finger_total, P_field_total, N_field_total, friction_cones_total = [], [], [], [], []
    for i in range(num_fingers):
        F_mask = np.linalg.norm(measurement[i]["Fn_field"], axis=1) > 0.03
        P_field, N_field, N_field_finger = np.array(measurement[i]["P_field"])[F_mask], np.array(measurement[i]["N_field"])[F_mask], np.array(measurement[i]["N_field_finger"])[F_mask]
        P_field_total.append(P_field)  # (N, 3): The position field of all contact points
        N_field_total.append(N_field)
        G, friction_cones = calculate_G_matrix(P_field, N_field, centroid, friction)
        G_finger, _ = calculate_G_matrix(P_field, N_field_finger, centroid, friction)
        friction_cones_total.append(friction_cones)  # (N * 8, 3): The friction cone vectors for all contact points
        G_total.append(G)
        G_finger_total.append(G_finger)
    P_field_total = np.concatenate(P_field_total, axis=0)  # (N, 3): The position field of all contact points
    P_field_total = P_field_total.reshape(1, -1, 3).transpose(1, 0, 2).repeat(8, axis=0).reshape(-1, 3)
    N_field_total = np.concatenate(N_field_total, axis=0)  # (N, 3): The normal field of all contact points
    N_field_total = N_field_total.reshape(1, -1, 3).transpose(1, 0, 2).repeat(8, axis=0).reshape(-1, 3)
    friction_cones_total = np.concatenate(friction_cones_total, axis=0)  # (N * 8, 3): The friction cone vectors for all contact points
    G_total = np.concatenate(G_total, axis=0)  # (N * 8, 6)
    G_finger_total = np.concatenate(G_finger_total, axis=0)  # (N * 8, 6)

    # Calculate the closure metric by solving the optimization problem
    from scipy.optimize import linprog
    from scipy.optimize import minimize
    G_a = G_total[:, [0, 1, 3, 4, 5]]  # (N * 8, 5): The equality constraint matrix
    G_f = G_total[:, [2]]  # (N * 8, 1): The inequality constraint matrix
    G_p = G_finger_total[:, [2]]  # (N * 8, 1): The power matrix: The sum of the relative z force is 1
    n = G_a.shape[0]  # a 的长度


    # # 初始猜测
    # x0 = np.ones(n + 1) / (n + 1)

    # # 等式约束
    # def eq1(x):  # G_f @ a - f = 0
    #     return G_f.T @ x[:-1] - x[-1]
    # def eq2(x):  # sum(a) = 1
    #     return G_p.T @ x[:-1] - 1
    # def eq3(x):  # G_a @ a = 0
    #     return G_a.T @ x[:-1]

    # constraints = [
    #     {'type': 'eq', 'fun': eq1},
    #     {'type': 'eq', 'fun': eq2},
    #     {'type': 'eq', 'fun': eq3}
    # ]

    # 目标函数：minimize -f
    c = np.zeros(n + 1)
    c[-1] = -1
    # 等式约束
    A_eq = np.vstack(
        [np.hstack([G_f.T, -np.ones((G_f.shape[1], 1))]),  # G_f @ a - f = 0
         np.hstack([G_p.T, np.zeros((1, 1))]),  # G_p @ a = 1
         np.hstack([G_a.T, np.zeros((G_a.shape[1], 1))])])  # G_a @ a = 0
    b_eq = np.concatenate(
        [np.zeros(G_f.shape[1]), 
         np.array([1]), 
         np.zeros(G_a.shape[1])])

    # 变量范围
    bounds = [(0, 16 / n)] * n + [(None, None)]  # a >= 0, f 无约束

    # # 求解
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    # res = minimize(closure_objective, x0, args=(0.01,), constraints=constraints, bounds=bounds, method='SLSQP')

    if res.success:
        a_opt = res.x[:-1]
        f_opt = res.x[-1]
        # print("最优解 f =", f_opt)
    else:
        f_opt = 0
        print("优化失败")
    
    if draw == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # 绘制实心箭头（quiver 默认就是实心的）
        ax.quiver(
            P_field_total[:, 0], P_field_total[:, 1], P_field_total[:, 2],
            a_opt*N_field_total[:, 0], a_opt*N_field_total[:, 1], a_opt*N_field_total[:, 2],
            length=0.01, normalize=False, color='b', alpha=0.3, arrow_length_ratio=0.3)
        ax.quiver(
            P_field_total[:, 0], P_field_total[:, 1], P_field_total[:, 2],
            a_opt*friction_cones_total[:, 0], a_opt*friction_cones_total[:, 1], a_opt*friction_cones_total[:, 2],
            length=0.01, normalize=False, color='r', arrow_length_ratio=0.3)
        ax.scatter3D(centroid[0], centroid[1], centroid[2], color='r', s=100, label='Centroid')  # 绘制物体质心
        # 加上 xyz 轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Normal Vectors')
        plt.show()

    return f_opt

def closure_objective(x, lambda_reg=1):
    a = x[:-1]
    f = x[-1]
    return -f + lambda_reg * np.linalg.norm(a)**2



def calculate_antipodal_metric(measurement, centroid=np.array([0, 0, 0])):
    num_fingers = len(measurement)
    metric = np.zeros(num_fingers)
    P_list, N_list, N_finger_list = [], [], []
    for i in range(num_fingers):
        F_mask = np.linalg.norm(measurement[i]["Fn_field"], axis=1) > 0.03
        P_field, N_field, N_field_finger = np.array(measurement[i]["P_field"]), np.array(measurement[i]["N_field"]), np.array(measurement[i]["N_field_finger"])
        P, N, N_finger = np.mean(P_field[F_mask], axis=0), np.mean(N_field[F_mask], axis=0), np.mean(N_field_finger[F_mask], axis=0)
        P_list.append(P)
        N_list.append(N)
        N_finger_list.append(N_finger)
        # print(f"Finger {i+1}: P: {P}, N: {N}, N_finger: {N_finger}")
    # the angle of N_list[0] and N_list[1]
    alpha = np.arccos(-np.dot(N_list[0], N_list[1]) / (np.linalg.norm(N_list[0]) * np.linalg.norm(N_list[1])))
    beta = np.arccos(np.dot(N_finger_list[0], np.array([0, 0, 1])) / (np.linalg.norm(N_finger_list[0]))) + np.arccos(np.dot(N_finger_list[1], np.array([0, 0, 1])) / (np.linalg.norm(N_finger_list[1])))
    metric = np.array([alpha, beta])
    distance = np.linalg.norm(np.mean(np.array(P_list)[:, :2], axis=0) - centroid[:2])

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
        # F_mask = np.linalg.norm(F_field_corrected, axis=1) > 0.03
        # ratio = np.linalg.norm(F_field_corrected[:, :2], axis=1)

        F_mask = np.linalg.norm(measurement[i]["Fn_field"], axis=1) > 0.03
        if F_mask.sum() == 0:
            metric[i], Fv[i] = 1.0, 0.0
            continue
        
        ratio = np.linalg.norm(measurement[i]["Ft_field"], axis=1) / (np.linalg.norm(measurement[i]["Fn_field"], axis=1) + 1e-6)
        
        metric[i] = sum(ratio[F_mask]) / (sum(F_mask))

        # metric[i] = np.sum(np.linalg.norm(measurement[i]["Ft_field"], axis=1)) / np.sum(np.linalg.norm(measurement[i]["Fn_field"], axis=1))

        Fv[i] = measurement[i]["Fv"]
        # print(f"Finger {i+1}: Metric: {sum(ratio[F_mask]) / sum(F_mask)}, Fv: {Fv[i]}")
    
    return metric, Fv


if __name__ == "__main__":
    N_field = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    P_field = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    centroid = np.array([0, 0, 0])
    G = calculate_G_matrix(P_field, N_field, centroid)
    print("G matrix:\n", G)