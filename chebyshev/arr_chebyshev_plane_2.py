import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import chebwin



############################################## 切比雪夫方法 -- 核心方法 #############################################

def compute_array_pattern(lambda_, d, N, M, theta0, phi0, weights, NN, MM, eps):
    """
    计算均匀矩形平面相控阵的方向图

    参数:
    - lambda_ : 波长
    - d : 单元间距
    - N : 水平方向上的单元数
    - M : 垂直方向上的单元数
    - theta0 : 波束指向角（俯仰角，弧度）
    - phi0 : 波束指向角（方位角，弧度）
    - weights : 激励权重（二维数组，大小为NxM）

    返回:
    - AF : 方向图（归一化后的幅度）
    - Theta : 俯仰角网格（弧度）
    - Phi : 方位角网格（弧度）
    """

    # 计算相位偏移
    kx = 2 * np.pi / lambda_ * np.sin(theta0) * np.cos(phi0)
    ky = 2 * np.pi / lambda_ * np.sin(theta0) * np.sin(phi0)

    # 计算theta和phi的范围
    theta = np.linspace(0, np.pi / 2, NN)
    phi = np.linspace(-np.pi, np.pi, MM)
    Theta, Phi = np.meshgrid(theta, phi)

    # 将theta和phi转换为u和v
    U = np.sin(Theta) * np.cos(Phi)
    V = np.sin(Theta) * np.sin(Phi)

    # 初始化方向图
    AF = np.zeros(U.shape, dtype=complex)

    # 计算阵列因子
    for n in range(N):
        for m in range(M):
            AF += weights[n, m] * np.exp(1j * 2 * np.pi / lambda_ * (n * d * U + m * d * V) - 1j * (n * kx + m * ky))

    # 取绝对值并归一化
    # AF = np.abs(AF)
    # AF = AF / np.max(AF)

    # 转dB
    pattern_dbw = 20 * np.log10(np.abs(AF) / np.max(np.abs(AF)) + eps)

    return pattern_dbw, Theta, Phi, theta, phi

################################################## 给外部调用 ###############################################
def arr_chebyshev_plane(lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps):
    theta0_rad = math.radians(theta0)
    phi0_rad = math.radians(phi0)
    #
    Ty = chebwin(Ny, abs(SLL))
    Tz = chebwin(Nz, abs(SLL))
    # 生成权重矩阵
    weights = np.outer(Ty, Tz)
    # 3.生成方向图
    pattern_dbw, Theta, Phi, theta, phi = compute_array_pattern(lambda_, d, Ny, Nz, theta0_rad, phi0_rad, weights, NA, NE, eps)
    #
    return weights, pattern_dbw, theta, phi
################################################## 画图相关 #################################################


def plot_array_pattern(AF, Theta, Phi):
    """
    绘制方向图

    参数:
    - AF : 方向图（归一化后的幅度）
    - Theta : 俯仰角网格（弧度）
    - Phi : 方位角网格（弧度）
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Theta * 180 / np.pi, Phi * 180 / np.pi, AF, cmap='viridis')

    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Phi (degrees)')
    ax.set_zlabel('Normalized AF')
    ax.set_title('Uniform Rectangular Planar Array Pattern with Beam Steering')

    plt.show()


def plot_array_pattern_theta_phi(AF, theta, phi):
    max_index = np.unravel_index(np.argmax(AF), AF.shape)
    print("max_index: %f, %f" % (max_index[0], max_index[1]))
    temp2_1 = AF[max_index[0], :]
    temp2_2 = AF[:, max_index[1]]
    plt.figure()
    plt.plot(theta * 180 / np.pi, temp2_1, label='temp2_1', color='#ff7f0e')
    plt.grid()
    plt.legend()
    plt.xlabel('theta (degree)')
    plt.ylabel('normalized pattern (dB)')
    plt.show()
    #
    plt.figure()
    plt.plot(phi * 180 / np.pi, temp2_2, label='temp2_2', color='#ff7f0e')
    plt.grid()
    plt.legend()
    plt.xlabel('phi (degree)')
    plt.ylabel('normalized pattern (dB)')
    plt.show()


###################################################### 测试函数 ####################################################
def test_point_weight_1():
    # 1.设置参数
    lambda_ = 1
    d = lambda_ / 2
    N = 40
    M = 40
    # 波束指向角 (theta0, phi0)
    # theta0 = 0  # 0度
    # phi0 = 0  # 0度
    theta0 = np.pi / 24  # 7.5度
    phi0 = np.pi / 4  # 45度
    # theta0 = np.pi / 12  # 15度
    # phi0 = np.pi / 4  # 45度
    #
    # 2.设置激励 -- 阵元激励全1
    weights = np.ones((N, M))  # 所有阵元的激励权重为1
    # 3.生成方向图
    AF, Theta, Phi, theta, phi = compute_array_pattern(lambda_, d, N, M, theta0, phi0, weights, 500, 500, 0.0001)
    # 4.画图检测
    plot_array_pattern(AF, Theta, Phi)
    plot_array_pattern_theta_phi(AF, theta, phi)


def test_point_weight_chebyshev():
    # 1.设置参数
    lambda_ = 1
    d = lambda_ / 2
    N = 40
    M = 40
    # 波束指向角 (theta0, phi0)
    # theta0 = 0  # 0度
    # phi0 = 0  # 0度
    # theta0 = np.pi / 24  # 7.5度
    # phi0 = np.pi / 4  # 45度
    # theta0 = np.pi / 12  # 15度
    # phi0 = np.pi / 4  # 45度
    theta0 = np.pi / 6  # 30度
    phi0 = 0  # 0度
    #
    # 2.设置激励 -- 切比雪夫方法
    SLL = -35
    Ty = chebwin(N, abs(SLL))
    Tz = chebwin(M, abs(SLL))
    # 生成权重矩阵
    weights = np.outer(Ty, Tz)
    # 3.生成方向图
    AF, Theta, Phi, theta, phi = compute_array_pattern(lambda_, d, N, M, theta0, phi0, weights, 500, 500, 0.0001)
    # 4.画图检测
    plot_array_pattern(AF, Theta, Phi)
    plot_array_pattern_theta_phi(AF, theta, phi)





if __name__ == '__main__':
    print('arr chebyshev plane 2')
    # 示例使用
    test_point_weight_1()
    # test_point_weight_chebyshev()
    #
    # 测试角度转弧度
    # angle_degrees = 15
    # angle_radians = math.radians(angle_degrees)
    #
    # print("角度：", angle_degrees)
    # print("弧度：", angle_radians)
    # print("15°弧度：", np.pi / 12)
    #
    # 测试调用
    # weights, pattern_dbw, theta, phi = arr_chebyshev_plane(lambda_=1, Ny=40, Nz=40, SLL=-60, d=0.5,
    #                                                        phi0=0, theta0=0, NA=360, NE=360, eps=0.0001)
    # Theta, Phi = np.meshgrid(theta, phi)
    # plot_array_pattern(pattern_dbw, Theta, Phi)
    # plot_array_pattern_theta_phi(pattern_dbw, theta, phi)
    # #
    # weights, pattern_dbw, theta, phi = arr_chebyshev_plane(lambda_=1, Ny=40, Nz=40, SLL=-60, d=0.5,
    #                                                        phi0=0, theta0=15, NA=360, NE=360, eps=0.0001)
    # Theta, Phi = np.meshgrid(theta, phi)
    # plot_array_pattern(pattern_dbw, Theta, Phi)
    # plot_array_pattern_theta_phi(pattern_dbw, theta, phi)

