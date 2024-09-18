import numpy as np
import matplotlib.pyplot as plt
import csv


def read_csv_to_2d_array_with_numbers(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 尝试将每列转换为浮点数，如果失败则保持为字符串
            processed_row = [float(element) if element.replace('.', '', 1).isdigit() else element for element in row]
            data.append(processed_row)
    return data


def pattern_dbw_from_weight(lambda_, N, NN, d, arr_w, theta0):
    # 计算波数
    k = 2 * np.pi / lambda_
    # 转换波束指向角到弧度
    steering_angle = np.radians(theta0)
    # 角度范围从-90度到90度（以弧度为单位）
    theta = np.linspace(-np.pi / 2, np.pi / 2, NN)
    # 初始化阵因子Array Factor(AF)
    AF = np.zeros(theta.shape, dtype=complex)
    # 计算阵因子
    for i, ang in enumerate(theta):
        elementPos = np.arange(N)  # 阵元位置矢量
        phaseDiff = k * d * (np.cos(np.pi / 2 - ang) - np.cos(np.pi / 2 - steering_angle))
        AF[i] = np.sum(arr_w * np.exp(1j * phaseDiff * elementPos))
    # 归一化阵因子
    # AF = np.abs(AF) / np.max(np.abs(AF))
    # AF转dB
    eps = 0.0001
    pattern_dbw = 20 * np.log10(np.abs(AF) / np.max(np.abs(AF)) + eps)
    # 转换角度到度
    theta_deg = np.rad2deg(theta)
    return pattern_dbw, theta_deg


def paper_img_compare_1(weight_CKM, weight_SWCKM, weight_WCKM, lambda_, N, d, theta0, NN):
    pattern_dbw_CKM, theta_CKM = pattern_dbw_from_weight(lambda_, N, NN, d, weight_CKM, theta0)
    pattern_dbw_SWCKM, theta_SWCKM = pattern_dbw_from_weight(lambda_, N, NN, d, weight_SWCKM, theta0)
    pattern_dbw_WCKM, theta_WCKM = pattern_dbw_from_weight(lambda_, N, NN, d, weight_WCKM, theta0)
    # 绘制方向图
    plt.figure()
    #
    plt.plot(theta_CKM, pattern_dbw_CKM, label='KCM', color='#ff7f0e')
    plt.plot(theta_SWCKM, pattern_dbw_SWCKM, label='SWKCM', color='#d62728')
    plt.plot(theta_WCKM, pattern_dbw_WCKM, label='WKCM', color='#9467bd')
    # plt.plot(phi * 180 / np.pi, temp1_NM, label='NM', color='#2ca02c')
    # plt.plot(phi * 180 / np.pi, temp1_NN, label='UPM', color='#1f77b4')
    plt.grid()
    plt.legend()
    plt.xlabel('theta (degree)')
    plt.ylabel('normalized pattern (dB)')
    # plt.title('Chebyshev Plane Array (15x15 elements, 15 subarray)')
    plt.show()


def paper_img_compare_2(weight_CKM_0, weight_SWCKM_0, weight_WCKM_0,
                        weight_CKM_15, weight_SWCKM_15, weight_WCKM_15,
                        lambda_, N, d, NN):
    pattern_dbw_CKM_0, theta_CKM_0 = pattern_dbw_from_weight(lambda_, N, NN, d, weight_CKM_0, 0)
    pattern_dbw_SWCKM_0, theta_SWCKM_0 = pattern_dbw_from_weight(lambda_, N, NN, d, weight_SWCKM_0, 0)
    pattern_dbw_WCKM_0, theta_WCKM_0 = pattern_dbw_from_weight(lambda_, N, NN, d, weight_WCKM_0, 0)
    pattern_dbw_CKM_15, theta_CKM_15 = pattern_dbw_from_weight(lambda_, N, NN, d, weight_CKM_15, 15)
    pattern_dbw_SWCKM_15, theta_SWCKM_15 = pattern_dbw_from_weight(lambda_, N, NN, d, weight_SWCKM_15, 15)
    pattern_dbw_WCKM_15, theta_WCKM_15 = pattern_dbw_from_weight(lambda_, N, NN, d, weight_WCKM_15, 15)
    # 绘制方向图
    plt.figure()
    #
    plt.plot(theta_CKM_0, pattern_dbw_CKM_0, label='KCM (0°)', color='#ff7f0e')
    plt.plot(theta_SWCKM_0, pattern_dbw_SWCKM_0, label='SWKCM (0°)', color='#d62728')
    plt.plot(theta_WCKM_0, pattern_dbw_WCKM_0, label='WKCM (0°)', color='#9467bd')
    # plt.plot(phi * 180 / np.pi, temp1_NM, label='NM', color='#2ca02c')
    # plt.plot(phi * 180 / np.pi, temp1_NN, label='UPM', color='#1f77b4')
    plt.plot(theta_CKM_15, pattern_dbw_CKM_15, label='KCM (15°)', color='#ff7f0e', linestyle='--')
    plt.plot(theta_SWCKM_15, pattern_dbw_SWCKM_15, label='SWKCM (15°)', color='#d62728', linestyle='--')
    plt.plot(theta_WCKM_15, pattern_dbw_WCKM_15, label='WKCM (15°)', color='#9467bd', linestyle='--')
    # plt.plot(phi * 180 / np.pi, temp1_NM, label='NM', color='#2ca02c')
    # plt.plot(phi * 180 / np.pi, temp1_NN, label='UPM', color='#1f77b4')
    plt.grid()
    plt.legend()
    plt.xlabel('theta (degree)')
    plt.ylabel('normalized pattern (dB)')
    # plt.title('Chebyshev Plane Array (15x15 elements, 15 subarray)')
    plt.show()


def drawWeight(data):
    plt.figure()
    plt.imshow(data)
    plt.axis('off')  # 这将隐藏x轴和y轴
    plt.show()




if __name__ == '__main__':
    # 【0. 初始化参数】
    lambda_ = 1  # 波长
    d = 0.5  # 阵元间隔
    N = 100  # 阵元个数
    eps = 0.0001  # 底电平
    NN = 1000  # 方位角度采样
    SLL = -35
    k_kmeans = 9  # 子阵数量

    theta0 = 15      # 波束指向角

    # 读取CSV文件
    # file_path_CKM = "./files/subarray/100-9/best_arr_w_kcm__(15)-2024-08-26-001.csv"
    # file_path_SWCKM = "./files/subarray/100-9/best_arr_w_swkcm__(15)-2024-08-26-001.csv"
    # file_path_WCKM = "./files/subarray/100-9/best_arr_w_wkcm__(15)-2024-08-26-001.csv"
    # file_path_NM = "./files/40x40-9-sll60/subarrayNM_40x40-9_60_2024-06-27.csv"
    # file_path_NN = "./files/40x40-9-sll60/subarrayNN_40x40-9_60_2024-06-28.csv"
    #
    # weight_CKM = read_csv_to_2d_array_with_numbers(file_path_CKM)
    # weight_SWCKM = read_csv_to_2d_array_with_numbers(file_path_SWCKM)
    # weight_WCKM = read_csv_to_2d_array_with_numbers(file_path_WCKM)
    # weight_NM = read_csv_to_2d_array_with_numbers(file_path_NM)
    # weight_NN = read_csv_to_2d_array_with_numbers(file_path_NN)
    # weight_NN = np.array(weight_NN)[1:41, 1:41]
    #
    #
    file_path_CKM_0 = "./files/subarray/100-9/best_arr_w_kcm__(0)-2024-08-26-001.csv"
    file_path_SWCKM_0 = "./files/subarray/100-9/best_arr_w_swkcm__(0)-2024-08-26-001.csv"
    file_path_WCKM_0 = "./files/subarray/100-9/best_arr_w_wkcm__(0)-2024-08-26-001.csv"
    file_path_CKM_15 = "./files/subarray/100-9/best_arr_w_kcm__(15)-2024-08-26-001.csv"
    file_path_SWCKM_15 = "./files/subarray/100-9/best_arr_w_swkcm__(15)-2024-08-26-001.csv"
    file_path_WCKM_15 = "./files/subarray/100-9/best_arr_w_wkcm__(15)-2024-08-26-001.csv"
    #
    weight_CKM_0 = read_csv_to_2d_array_with_numbers(file_path_CKM_0)
    weight_SWCKM_0 = read_csv_to_2d_array_with_numbers(file_path_SWCKM_0)
    weight_WCKM_0 = read_csv_to_2d_array_with_numbers(file_path_WCKM_0)
    weight_CKM_15 = read_csv_to_2d_array_with_numbers(file_path_CKM_15)
    weight_SWCKM_15 = read_csv_to_2d_array_with_numbers(file_path_SWCKM_15)
    weight_WCKM_15 = read_csv_to_2d_array_with_numbers(file_path_WCKM_15)
    #
    #
    # 论文图 -- 单图子阵划分结果
    # drawWeight(weight_CKM)
    # drawWeight(weight_SWCKM)
    # drawWeight(weight_WCKM)
    # drawWeight(weight_NM)
    # drawWeight(weight_NN)
    #
    #
    #
    # 论文图 -- 比较
    # paper_img_compare_1(weight_CKM, weight_SWCKM, weight_WCKM, lambda_, N, d, theta0, NN)
    paper_img_compare_2(weight_CKM_0, weight_SWCKM_0, weight_WCKM_0,
                        weight_CKM_15, weight_SWCKM_15, weight_WCKM_15,
                        lambda_, N, d, NN)