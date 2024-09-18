import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.font_manager import FontProperties
from subarray.ArrayChebyshev import ArrayChebyshev
import chebyshev.arr_chebyshev_plane as arr_chebyshev_plane


def read_csv_to_2d_array_with_numbers(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 尝试将每列转换为浮点数，如果失败则保持为字符串
            processed_row = [float(element) if element.replace('.', '', 1).isdigit() else element for element in row]
            data.append(processed_row)
    return data


def average_values_by_indices(weights_indices):
    arrayCheby = ArrayChebyshev()
    weights = arrayCheby.getChebyshevPlane(40, 40, -30)
    # 初始化weights_sub为weights的形状，初始填充NaN以方便后续检查是否所有索引都被处理
    weights_sub = np.full_like(weights, np.nan)
    # 遍历所有可能的索引值
    unique_indices = np.unique(weights_indices)
    for index in unique_indices:
        # 找到weights_indices中等于当前index的所有位置
        mask = weights_indices == index
        # 根据位置mask选择weights中的值，然后计算这些值的平均值
        avg_value = np.mean(weights[mask])
        # 将平均值赋给weights_sub中对应的位置
        weights_sub[mask] = avg_value
    # 检查weights_sub中是否还有NaN值，确保所有位置都被正确处理
    if np.isnan(weights_sub).any():
        raise ValueError("有些索引可能没有对应的值进行平均，请检查weights_indices是否包含了所有必要的索引。")
    return weights_sub


def paper_img_compare(weight_CKM, weight_SWCKM, weight_NM, weight_GANM, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps):
    phi = np.linspace(-np.pi / 2, np.pi / 2, NA)
    theta = np.linspace(-np.pi / 2, np.pi / 2, NE)
    aa = np.arange(0, d * Ny, d)
    bb = np.arange(0, d * Nz, d)
    DD1 = np.repeat(aa[:, np.newaxis], Nz, axis=1)
    DD2 = np.repeat(bb[np.newaxis, :], Ny, axis=0)
    #
    pattern_CKM = np.zeros((len(phi), len(theta)), dtype=complex)
    pattern_SWCKM = np.zeros((len(phi), len(theta)), dtype=complex)
    pattern_NM = np.zeros((len(phi), len(theta)), dtype=complex)
    pattern_GANM = np.zeros((len(phi), len(theta)), dtype=complex)
    #
    for jj in range(len(phi)):
        for ii in range(len(theta)):
            pattern0_CKM = weight_CKM * np.exp(1j * 2 * np.pi / lambda_ *
                                        (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                         np.sin(theta[ii]) * DD2 -
                                         np.sin(phi0) * np.cos(theta0) * DD1 -
                                         np.sin(theta0) * DD2))
            pattern_CKM[jj, ii] = np.sum(pattern0_CKM)
            pattern0_SWCKM = weight_SWCKM * np.exp(1j * 2 * np.pi / lambda_ *
                                            (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                             np.sin(theta[ii]) * DD2 -
                                             np.sin(phi0) * np.cos(theta0) * DD1 -
                                             np.sin(theta0) * DD2))
            pattern_SWCKM[jj, ii] = np.sum(pattern0_SWCKM)
            pattern0_NM = weight_NM * np.exp(1j * 2 * np.pi / lambda_ *
                                               (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                                np.sin(theta[ii]) * DD2 -
                                                np.sin(phi0) * np.cos(theta0) * DD1 -
                                                np.sin(theta0) * DD2))
            pattern_NM[jj, ii] = np.sum(pattern0_NM)
            pattern0_GANM = weight_GANM * np.exp(1j * 2 * np.pi / lambda_ *
                                                   (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                                    np.sin(theta[ii]) * DD2 -
                                                    np.sin(phi0) * np.cos(theta0) * DD1 -
                                                    np.sin(theta0) * DD2))
            pattern_GANM[jj, ii] = np.sum(pattern0_GANM)
    #
    max_p_CKM = np.max(np.abs(pattern_CKM))
    pattern_dbw_CKM = 20 * np.log10(np.abs(pattern_CKM) / max_p_CKM + eps)
    max_p_SWCKM = np.max(np.abs(pattern_SWCKM))
    pattern_dbw_SWCKM = 20 * np.log10(np.abs(pattern_SWCKM) / max_p_SWCKM + eps)
    max_p_NM = np.max(np.abs(pattern_NM))
    pattern_dbw_NM = 20 * np.log10(np.abs(pattern_NM) / max_p_NM + eps)
    max_p_GANM = np.max(np.abs(pattern_GANM))
    pattern_dbw_GANM = 20 * np.log10(np.abs(pattern_GANM) / max_p_GANM + eps)
    #
    # 绘制方向图
    pattern_dbw_CKM[pattern_dbw_CKM < -50] = -50 + np.random.uniform(-1, 1, np.sum(pattern_dbw_CKM < -50))
    pattern_dbw_SWCKM[pattern_dbw_SWCKM < -50] = -50 + np.random.uniform(-1, 1, np.sum(pattern_dbw_SWCKM < -50))
    pattern_dbw_NM[pattern_dbw_NM < -50] = -50 + np.random.uniform(-1, 1, np.sum(pattern_dbw_NM < -50))
    pattern_dbw_GANM[pattern_dbw_GANM < -50] = -50 + np.random.uniform(-1, 1, np.sum(pattern_dbw_GANM < -50))
    # pattern_dbw_CKM[pattern_dbw_CKM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_CKM < -40))
    # pattern_dbw_SWCKM[pattern_dbw_SWCKM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_SWCKM < -40))
    # pattern_dbw_NM[pattern_dbw_NM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_NM < -40))
    # pattern_dbw_GANM[pattern_dbw_GANM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_GANM < -40))
    # 绘制方位向切面图
    plt.figure()
    temp1_CKM = pattern_dbw_CKM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp1_SWCKM = pattern_dbw_SWCKM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp1_NM = pattern_dbw_NM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp1_GANM = pattern_dbw_GANM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    #
    phi2 = phi[180:]
    temp1_CKM_2 = temp1_CKM[180:]
    temp1_SWCKM_2 = temp1_SWCKM[180:]
    temp1_NM_2 = temp1_NM[180:]
    temp1_GANM_2 = temp1_GANM[180:]
    plt.plot(phi2 * 180 / np.pi, temp1_CKM_2, label='KCM', color='blue')
    plt.plot(phi2 * 180 / np.pi, temp1_SWCKM_2, label='SWKCM', color='red')
    plt.plot(phi2 * 180 / np.pi, temp1_NM_2, label='NM', color='black')
    plt.plot(phi2 * 180 / np.pi, temp1_GANM_2, label='GANM', color='green')
    # plt.plot(phi * 180 / np.pi, temp1_CKM, label='KCM', color='blue')
    # plt.plot(phi * 180 / np.pi, temp1_SWCKM, label='SWKCM', color='red')
    # plt.plot(phi * 180 / np.pi, temp1_NM, label='NM', color='black')
    # plt.plot(phi * 180 / np.pi, temp1_GANM, label='GANM', color='green')
    plt.grid()
    plt.legend()
    plt.xlabel('phi (degree)')
    plt.ylabel('normalized pattern (dB)')
    # plt.title('Chebyshev Plane Array (15x15 elements, 15 subarray)')
    plt.show()


def paper_img_compare_2(weight_CKM, weight_SWCKM, weight_NM, weight_NN, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps):
    phi = np.linspace(-np.pi / 2, np.pi / 2, NA)
    theta = np.linspace(-np.pi / 2, np.pi / 2, NE)
    aa = np.arange(0, d * Ny, d)
    bb = np.arange(0, d * Nz, d)
    DD1 = np.repeat(aa[:, np.newaxis], Nz, axis=1)
    DD2 = np.repeat(bb[np.newaxis, :], Ny, axis=0)
    #
    pattern_CKM = np.zeros((len(phi), len(theta)), dtype=complex)
    pattern_SWCKM = np.zeros((len(phi), len(theta)), dtype=complex)
    pattern_NM = np.zeros((len(phi), len(theta)), dtype=complex)
    pattern_NN = np.zeros((len(phi), len(theta)), dtype=complex)
    #
    for jj in range(len(phi)):
        for ii in range(len(theta)):
            pattern0_CKM = weight_CKM * np.exp(1j * 2 * np.pi / lambda_ *
                                        (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                         np.sin(theta[ii]) * DD2 -
                                         np.sin(phi0) * np.cos(theta0) * DD1 -
                                         np.sin(theta0) * DD2))
            pattern_CKM[jj, ii] = np.sum(pattern0_CKM)
            pattern0_SWCKM = weight_SWCKM * np.exp(1j * 2 * np.pi / lambda_ *
                                            (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                             np.sin(theta[ii]) * DD2 -
                                             np.sin(phi0) * np.cos(theta0) * DD1 -
                                             np.sin(theta0) * DD2))
            pattern_SWCKM[jj, ii] = np.sum(pattern0_SWCKM)
            pattern0_NM = weight_NM * np.exp(1j * 2 * np.pi / lambda_ *
                                               (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                                np.sin(theta[ii]) * DD2 -
                                                np.sin(phi0) * np.cos(theta0) * DD1 -
                                                np.sin(theta0) * DD2))
            pattern_NM[jj, ii] = np.sum(pattern0_NM)
            pattern0_NN = weight_NN * np.exp(1j * 2 * np.pi / lambda_ *
                                                   (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                                    np.sin(theta[ii]) * DD2 -
                                                    np.sin(phi0) * np.cos(theta0) * DD1 -
                                                    np.sin(theta0) * DD2))
            pattern_NN[jj, ii] = np.sum(pattern0_NN)
    #
    max_p_CKM = np.max(np.abs(pattern_CKM))
    pattern_dbw_CKM = 20 * np.log10(np.abs(pattern_CKM) / max_p_CKM + eps)
    max_p_SWCKM = np.max(np.abs(pattern_SWCKM))
    pattern_dbw_SWCKM = 20 * np.log10(np.abs(pattern_SWCKM) / max_p_SWCKM + eps)
    max_p_NM = np.max(np.abs(pattern_NM))
    pattern_dbw_NM = 20 * np.log10(np.abs(pattern_NM) / max_p_NM + eps)
    max_p_NN = np.max(np.abs(pattern_NN))
    pattern_dbw_NN = 20 * np.log10(np.abs(pattern_NN) / max_p_NN + eps)
    #
    # 绘制方向图
    pattern_dbw_CKM[pattern_dbw_CKM < -80] = -80 + np.random.uniform(-1, 1, np.sum(pattern_dbw_CKM < -80))
    pattern_dbw_SWCKM[pattern_dbw_SWCKM < -80] = -80 + np.random.uniform(-1, 1, np.sum(pattern_dbw_SWCKM < -80))
    pattern_dbw_NM[pattern_dbw_NM < -80] = -80 + np.random.uniform(-1, 1, np.sum(pattern_dbw_NM < -80))
    pattern_dbw_NN[pattern_dbw_NN < -80] = -80 + np.random.uniform(-1, 1, np.sum(pattern_dbw_NN < -80))
    # pattern_dbw_CKM[pattern_dbw_CKM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_CKM < -40))
    # pattern_dbw_SWCKM[pattern_dbw_SWCKM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_SWCKM < -40))
    # pattern_dbw_NM[pattern_dbw_NM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_NM < -40))
    # pattern_dbw_NN[pattern_dbw_NN < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_NN < -40))
    # 绘制方位向切面图
    plt.figure()
    temp1_CKM = pattern_dbw_CKM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp1_SWCKM = pattern_dbw_SWCKM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp1_NM = pattern_dbw_NM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp1_NN = pattern_dbw_NN[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    #
    phi2 = phi[180:]
    temp1_CKM_2 = temp1_CKM[180:]
    temp1_SWCKM_2 = temp1_SWCKM[180:]
    temp1_NM_2 = temp1_NM[180:]
    temp1_NN_2 = temp1_NN[180:]
    plt.plot(phi2 * 180 / np.pi, temp1_CKM_2, label='KCM', color='blue')
    plt.plot(phi2 * 180 / np.pi, temp1_SWCKM_2, label='SWKCM', color='red')
    plt.plot(phi2 * 180 / np.pi, temp1_NM_2, label='NM', color='green')
    plt.plot(phi2 * 180 / np.pi, temp1_NN_2, label='UPM', color='black')
    # plt.plot(phi * 180 / np.pi, temp1_CKM, label='KCM', color='blue')
    # plt.plot(phi * 180 / np.pi, temp1_SWCKM, label='SWKCM', color='red')
    # plt.plot(phi * 180 / np.pi, temp1_NM, label='NM', color='black')
    # plt.plot(phi * 180 / np.pi, temp1_NN, label='GANM', color='green')
    plt.grid()
    plt.legend()
    plt.xlabel('phi (degree)')
    plt.ylabel('normalized pattern (dB)')
    # plt.title('Chebyshev Plane Array (15x15 elements, 15 subarray)')
    plt.show()


def paper_img_compare_3(weight_CKM, weight_SWCKM, weight_WCKM, weight_NM, weight_NN, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps):
    phi = np.linspace(-np.pi / 2, np.pi / 2, NA)
    theta = np.linspace(-np.pi / 2, np.pi / 2, NE)
    aa = np.arange(0, d * Ny, d)
    bb = np.arange(0, d * Nz, d)
    DD1 = np.repeat(aa[:, np.newaxis], Nz, axis=1)
    DD2 = np.repeat(bb[np.newaxis, :], Ny, axis=0)
    #
    pattern_CKM = np.zeros((len(phi), len(theta)), dtype=complex)
    pattern_SWCKM = np.zeros((len(phi), len(theta)), dtype=complex)
    pattern_WCKM = np.zeros((len(phi), len(theta)), dtype=complex)
    pattern_NM = np.zeros((len(phi), len(theta)), dtype=complex)
    pattern_NN = np.zeros((len(phi), len(theta)), dtype=complex)
    #
    for jj in range(len(phi)):
        for ii in range(len(theta)):
            pattern0_CKM = weight_CKM * np.exp(1j * 2 * np.pi / lambda_ *
                                        (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                         np.sin(theta[ii]) * DD2 -
                                         np.sin(phi0) * np.cos(theta0) * DD1 -
                                         np.sin(theta0) * DD2))
            pattern_CKM[jj, ii] = np.sum(pattern0_CKM)
            pattern0_SWCKM = weight_SWCKM * np.exp(1j * 2 * np.pi / lambda_ *
                                            (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                             np.sin(theta[ii]) * DD2 -
                                             np.sin(phi0) * np.cos(theta0) * DD1 -
                                             np.sin(theta0) * DD2))
            pattern_SWCKM[jj, ii] = np.sum(pattern0_SWCKM)
            pattern0_WCKM = weight_WCKM * np.exp(1j * 2 * np.pi / lambda_ *
                                                   (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                                    np.sin(theta[ii]) * DD2 -
                                                    np.sin(phi0) * np.cos(theta0) * DD1 -
                                                    np.sin(theta0) * DD2))
            pattern_WCKM[jj, ii] = np.sum(pattern0_WCKM)
            pattern0_NM = weight_NM * np.exp(1j * 2 * np.pi / lambda_ *
                                               (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                                np.sin(theta[ii]) * DD2 -
                                                np.sin(phi0) * np.cos(theta0) * DD1 -
                                                np.sin(theta0) * DD2))
            pattern_NM[jj, ii] = np.sum(pattern0_NM)
            pattern0_NN = weight_NN * np.exp(1j * 2 * np.pi / lambda_ *
                                                   (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                                    np.sin(theta[ii]) * DD2 -
                                                    np.sin(phi0) * np.cos(theta0) * DD1 -
                                                    np.sin(theta0) * DD2))
            pattern_NN[jj, ii] = np.sum(pattern0_NN)
    #
    max_p_CKM = np.max(np.abs(pattern_CKM))
    pattern_dbw_CKM = 20 * np.log10(np.abs(pattern_CKM) / max_p_CKM + eps)
    max_p_SWCKM = np.max(np.abs(pattern_SWCKM))
    pattern_dbw_SWCKM = 20 * np.log10(np.abs(pattern_SWCKM) / max_p_SWCKM + eps)
    max_p_WCKM = np.max(np.abs(pattern_WCKM))
    pattern_dbw_WCKM = 20 * np.log10(np.abs(pattern_WCKM) / max_p_WCKM + eps)
    max_p_NM = np.max(np.abs(pattern_NM))
    pattern_dbw_NM = 20 * np.log10(np.abs(pattern_NM) / max_p_NM + eps)
    max_p_NN = np.max(np.abs(pattern_NN))
    pattern_dbw_NN = 20 * np.log10(np.abs(pattern_NN) / max_p_NN + eps)
    #
    # 绘制方向图
    pattern_dbw_CKM[pattern_dbw_CKM < -80] = -80 + np.random.uniform(-1, 1, np.sum(pattern_dbw_CKM < -80))
    pattern_dbw_SWCKM[pattern_dbw_SWCKM < -80] = -80 + np.random.uniform(-1, 1, np.sum(pattern_dbw_SWCKM < -80))
    pattern_dbw_WCKM[pattern_dbw_WCKM < -80] = -80 + np.random.uniform(-1, 1, np.sum(pattern_dbw_WCKM < -80))
    pattern_dbw_NM[pattern_dbw_NM < -80] = -80 + np.random.uniform(-1, 1, np.sum(pattern_dbw_NM < -80))
    pattern_dbw_NN[pattern_dbw_NN < -80] = -80 + np.random.uniform(-1, 1, np.sum(pattern_dbw_NN < -80))
    # pattern_dbw_CKM[pattern_dbw_CKM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_CKM < -40))
    # pattern_dbw_SWCKM[pattern_dbw_SWCKM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_SWCKM < -40))
    # pattern_dbw_WCKM[pattern_dbw_WCKM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_WCKM < -40))
    # pattern_dbw_NM[pattern_dbw_NM < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_NM < -40))
    # pattern_dbw_NN[pattern_dbw_NN < -40] = -40 + np.random.uniform(-1, 1, np.sum(pattern_dbw_NN < -40))
    # 绘制方位向切面图
    plt.figure()
    temp1_CKM = pattern_dbw_CKM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp1_SWCKM = pattern_dbw_SWCKM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp1_WCKM = pattern_dbw_WCKM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp1_NM = pattern_dbw_NM[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp1_NN = pattern_dbw_NN[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    #
    phi2 = phi[180:]
    temp1_CKM_2 = temp1_CKM[180:]
    temp1_SWCKM_2 = temp1_SWCKM[180:]
    temp1_WCKM_2 = temp1_WCKM[180:]
    temp1_NM_2 = temp1_NM[180:]
    temp1_NN_2 = temp1_NN[180:]
    plt.plot(phi2 * 180 / np.pi, temp1_CKM_2, label='KCM', color='#ff7f0e')
    plt.plot(phi2 * 180 / np.pi, temp1_SWCKM_2, label='SWKCM', color='#d62728')
    plt.plot(phi2 * 180 / np.pi, temp1_WCKM_2, label='WKCM', color='#9467bd')
    plt.plot(phi2 * 180 / np.pi, temp1_NM_2, label='NM', color='#2ca02c')
    plt.plot(phi2 * 180 / np.pi, temp1_NN_2, label='UPM', color='#1f77b4')
    # plt.plot(phi * 180 / np.pi, temp1_CKM, label='KCM', color='blue')
    # plt.plot(phi * 180 / np.pi, temp1_SWCKM, label='SWKCM', color='red')
    # plt.plot(phi * 180 / np.pi, temp1_WCKM, label='WKCM', color='magenta')
    # plt.plot(phi * 180 / np.pi, temp1_NM, label='NM', color='black')
    # plt.plot(phi * 180 / np.pi, temp1_NN, label='GANM', color='green')
    plt.grid()
    plt.legend()
    plt.xlabel('phi (degree)')
    plt.ylabel('normalized pattern (dB)')
    # plt.title('Chebyshev Plane Array (15x15 elements, 15 subarray)')
    plt.show()


def paper_img_compare_point_item(weight, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps):
    pattern_dbw, theta, phi = arr_chebyshev_plane.compute_array_pattern(lambda_, d, Ny, Nz, theta0, phi0, weight, NA, NE, eps)
    max_index = np.unravel_index(np.argmax(pattern_dbw), pattern_dbw.shape)
    pattern_dbw_phi = pattern_dbw[:, max_index[1]]
    pattern_dbw_theta = pattern_dbw[max_index[0], :]
    return pattern_dbw, theta, phi, pattern_dbw_theta, pattern_dbw_phi


def paper_img_compare_point(weight_CKM, weight_SWCKM, weight_WCKM, weight_NM, weight_NN,
                            lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps,
                            labels, colors, linewidths, linestyles):
    pattern_dbw_CKM, theta_CKM, phi_CKM, pattern_dbw_theta_CKM, pattern_dbw_phi_CKM = paper_img_compare_point_item(
        weight_CKM, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps)
    pattern_dbw_SWCKM, theta_SWCKM, phi_SWCKM, pattern_dbw_theta_SWCKM, pattern_dbw_phi_SWCKM = paper_img_compare_point_item(
        weight_SWCKM, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps)
    pattern_dbw_WCKM, theta_WCKM, phi_WCKM, pattern_dbw_theta_WCKM, pattern_dbw_phi_WCKM = paper_img_compare_point_item(
        weight_WCKM, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps)
    pattern_dbw_NM, theta_NM, phi_NM, pattern_dbw_theta_NM, pattern_dbw_phi_NM = paper_img_compare_point_item(
        weight_NM, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps)
    pattern_dbw_NN, theta_NN, phi_NN, pattern_dbw_theta_NN, pattern_dbw_phi_NN = paper_img_compare_point_item(
        weight_NN, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps)
    #
    plt.figure()
    #
    plt.plot(theta_NN * 180 / np.pi, pattern_dbw_theta_NN, label=labels[0], color=colors[0], linewidth=linewidths[0], linestyle=linestyles[0])
    plt.plot(theta_NM * 180 / np.pi, pattern_dbw_theta_NM, label=labels[1], color=colors[1], linewidth=linewidths[1], linestyle=linestyles[1])
    plt.plot(theta_CKM * 180 / np.pi, pattern_dbw_theta_CKM, label=labels[2], color=colors[2], linewidth=linewidths[2], linestyle=linestyles[2])
    plt.plot(theta_WCKM * 180 / np.pi, pattern_dbw_theta_WCKM, label=labels[3], color=colors[3], linewidth=linewidths[3], linestyle=linestyles[3])
    plt.plot(theta_SWCKM * 180 / np.pi, pattern_dbw_theta_SWCKM, label=labels[4], color=colors[4], linewidth=linewidths[4], linestyle=linestyles[4])
    plt.grid()
    plt.legend(ncol=2, loc='upper right', fontsize=24)  # 图例位于上方右侧
    plt.xlabel('theta (degree)', fontsize=24)
    plt.ylabel('normalized pattern (dB)', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()
    #
    plt.plot(phi_NN * 180 / np.pi, pattern_dbw_phi_NN, label=labels[0], color=colors[0], linewidth=linewidths[0], linestyle=linestyles[0])
    plt.plot(phi_NM * 180 / np.pi, pattern_dbw_phi_NM, label=labels[1], color=colors[1], linewidth=linewidths[1], linestyle=linestyles[1])
    plt.plot(phi_CKM * 180 / np.pi, pattern_dbw_phi_CKM, label=labels[2], color=colors[2], linewidth=linewidths[2], linestyle=linestyles[2])
    plt.plot(phi_WCKM * 180 / np.pi, pattern_dbw_phi_WCKM, label=labels[3], color=colors[3], linewidth=linewidths[3], linestyle=linestyles[3])
    plt.plot(phi_SWCKM * 180 / np.pi, pattern_dbw_phi_SWCKM, label=labels[4], color=colors[4], linewidth=linewidths[4], linestyle=linestyles[4])
    #
    plt.grid()
    plt.legend(ncol=2, loc='upper right', fontsize=24)  # 图例位于上方右侧
    plt.xlabel('theta (degree)', fontsize=24)
    plt.ylabel('normalized pattern (dB)', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # plt.title('Chebyshev Plane Array (15x15 elements, 15 subarray)')
    plt.show()


def paper_img_compare_point_3(weight_CKM_0, weight_SWCKM_0, weight_WCKM_0, weight_NM_0, weight_NN_0, phi0_0, theta0_0,
                              weight_CKM_15, weight_SWCKM_15, weight_WCKM_15, weight_NM_15, weight_NN_15, phi0_15, theta0_15,
                              weight_CKM_30, weight_SWCKM_30, weight_WCKM_30, weight_NM_30, weight_NN_30, phi0_30, theta0_30,
                              lambda_, Ny, Nz, d, NA, NE, eps):
    # Sample data (replace these with your actual data)
    theta_CKM_0 = np.linspace(-np.pi, np.pi, 100)
    pattern_dbw_theta_CKM_0 = np.random.randn(100)
    theta_SWCKM_0 = np.linspace(-np.pi, np.pi, 100)
    pattern_dbw_theta_SWCKM_0 = np.random.randn(100)
    theta_WCKM_0 = np.linspace(-np.pi, np.pi, 100)
    pattern_dbw_theta_WCKM_0 = np.random.randn(100)
    theta_NM_0 = np.linspace(-np.pi, np.pi, 100)
    pattern_dbw_theta_NM_0 = np.random.randn(100)
    theta_NN_0 = np.linspace(-np.pi, np.pi, 100)
    pattern_dbw_theta_NN_0 = np.random.randn(100)
    theta_CKM_30 = np.linspace(-np.pi, np.pi, 100)
    pattern_dbw_theta_CKM_30 = np.random.randn(100)
    theta_SWCKM_30 = np.linspace(-np.pi, np.pi, 100)
    pattern_dbw_theta_SWCKM_30 = np.random.randn(100)
    theta_WCKM_30 = np.linspace(-np.pi, np.pi, 100)
    pattern_dbw_theta_WCKM_30 = np.random.randn(100)
    theta_NM_30 = np.linspace(-np.pi, np.pi, 100)
    pattern_dbw_theta_NM_30 = np.random.randn(100)
    theta_NN_30 = np.linspace(-np.pi, np.pi, 100)
    pattern_dbw_theta_NN_30 = np.random.randn(100)

    plt.figure()

    # Plotting with thicker lines
    plt.plot(theta_NN_0 * 180 / np.pi, pattern_dbw_theta_NN_0, label='UPM (0°)', color='#1f77b4', linewidth=4)
    plt.plot(theta_NM_0 * 180 / np.pi, pattern_dbw_theta_NM_0, label='NM (0°)', color='#2ca02c', linewidth=4)
    plt.plot(theta_CKM_0 * 180 / np.pi, pattern_dbw_theta_CKM_0, label='KCM (0°)', color='#ff7f0e', linewidth=4)
    plt.plot(theta_WCKM_0 * 180 / np.pi, pattern_dbw_theta_WCKM_0, label='WKCM (0°)', color='#9467bd', linewidth=4)
    plt.plot(theta_SWCKM_0 * 180 / np.pi, pattern_dbw_theta_SWCKM_0, label='SWKCM (0°)', color='#d62728', linewidth=4)
    plt.plot(theta_NM_30 * 180 / np.pi, pattern_dbw_theta_NM_30, label='NM (30°)', color='#2ca02c', linestyle='--', linewidth=4.5)
    plt.plot(theta_NN_30 * 180 / np.pi, pattern_dbw_theta_NN_30, label='UPM (30°)', color='#1f77b4', linestyle='--', linewidth=4.5)
    plt.plot(theta_CKM_30 * 180 / np.pi, pattern_dbw_theta_CKM_30, label='KCM (30°)', color='#ff7f0e', linestyle='--', linewidth=4.5)
    plt.plot(theta_WCKM_30 * 180 / np.pi, pattern_dbw_theta_WCKM_30, label='WKCM (30°)', color='#9467bd', linestyle='--', linewidth=4.5)
    plt.plot(theta_SWCKM_30 * 180 / np.pi, pattern_dbw_theta_SWCKM_30, label='SWKCM (30°)', color='#d62728', linestyle='--', linewidth=4.5)

    # Making the grid, legend, labels bold
    plt.grid()
    plt.legend(ncol=2, loc='upper right', fontsize=24)  # 图例位于上方右侧
    plt.xlabel('theta (degree)', fontsize=24)
    plt.ylabel('normalized pattern (dB)', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()


def paper_img_compare_point_3_bak(weight_CKM_0, weight_SWCKM_0, weight_WCKM_0, weight_NM_0, weight_NN_0, phi0_0, theta0_0,
                              weight_CKM_15, weight_SWCKM_15, weight_WCKM_15, weight_NM_15, weight_NN_15, phi0_15, theta0_15,
                              weight_CKM_30, weight_SWCKM_30, weight_WCKM_30, weight_NM_30, weight_NN_30, phi0_30, theta0_30,
                              lambda_, Ny, Nz, d, NA, NE, eps):
    pattern_dbw_CKM_0, theta_CKM_0, phi_CKM_0, pattern_dbw_theta_CKM_0, pattern_dbw_phi_CKM_0 = paper_img_compare_point_item(
        weight_CKM_0, lambda_, Ny, Nz, d, math.radians(phi0_0), math.radians(theta0_0), NA, NE, eps)
    pattern_dbw_SWCKM_0, theta_SWCKM_0, phi_SWCKM_0, pattern_dbw_theta_SWCKM_0, pattern_dbw_phi_SWCKM_0 = paper_img_compare_point_item(
        weight_SWCKM_0, lambda_, Ny, Nz, d, math.radians(phi0_0), math.radians(theta0_0), NA, NE, eps)
    pattern_dbw_WCKM_0, theta_WCKM_0, phi_WCKM_0, pattern_dbw_theta_WCKM_0, pattern_dbw_phi_WCKM_0 = paper_img_compare_point_item(
        weight_WCKM_0, lambda_, Ny, Nz, d, math.radians(phi0_0), math.radians(theta0_0), NA, NE, eps)
    pattern_dbw_NM_0, theta_NM_0, phi_NM_0, pattern_dbw_theta_NM_0, pattern_dbw_phi_NM_0 = paper_img_compare_point_item(
        weight_NM_0, lambda_, Ny, Nz, d, math.radians(phi0_0), math.radians(theta0_0), NA, NE, eps)
    pattern_dbw_NN_0, theta_NN_0, phi_NN_0, pattern_dbw_theta_NN_0, pattern_dbw_phi_NN_0 = paper_img_compare_point_item(
        weight_NN_0, lambda_, Ny, Nz, d, math.radians(phi0_0), math.radians(theta0_0), NA, NE, eps)
    #
    # pattern_dbw_CKM_15, theta_CKM_15, phi_CKM_15, pattern_dbw_theta_CKM_15, pattern_dbw_phi_CKM_15 = paper_img_compare_point_item(
    #     weight_CKM_15, lambda_, Ny, Nz, d, math.radians(phi0_15), math.radians(theta0_15), NA, NE, eps)
    # pattern_dbw_SWCKM_15, theta_SWCKM_15, phi_SWCKM_15, pattern_dbw_theta_SWCKM_15, pattern_dbw_phi_SWCKM_15 = paper_img_compare_point_item(
    #     weight_SWCKM_15, lambda_, Ny, Nz, d, math.radians(phi0_15), math.radians(theta0_15), NA, NE, eps)
    # pattern_dbw_WCKM_15, theta_WCKM_15, phi_WCKM_15, pattern_dbw_theta_WCKM_15, pattern_dbw_phi_WCKM_15 = paper_img_compare_point_item(
    #     weight_WCKM_15, lambda_, Ny, Nz, d, math.radians(phi0_15), math.radians(theta0_15), NA, NE, eps)
    # pattern_dbw_NM_15, theta_NM_15, phi_NM_15, pattern_dbw_theta_NM_15, pattern_dbw_phi_NM_15 = paper_img_compare_point_item(
    #     weight_NM_15, lambda_, Ny, Nz, d, math.radians(phi0_15), math.radians(theta0_15), NA, NE, eps)
    # pattern_dbw_NN_15, theta_NN_15, phi_NN_15, pattern_dbw_theta_NN_15, pattern_dbw_phi_NN_15 = paper_img_compare_point_item(
    #     weight_NN_15, lambda_, Ny, Nz, d, math.radians(phi0_15), math.radians(theta0_15), NA, NE, eps)
    #
    pattern_dbw_CKM_30, theta_CKM_30, phi_CKM_30, pattern_dbw_theta_CKM_30, pattern_dbw_phi_CKM_30 = paper_img_compare_point_item(
        weight_CKM_30, lambda_, Ny, Nz, d, math.radians(phi0_30), math.radians(theta0_30), NA, NE, eps)
    pattern_dbw_SWCKM_30, theta_SWCKM_30, phi_SWCKM_30, pattern_dbw_theta_SWCKM_30, pattern_dbw_phi_SWCKM_30 = paper_img_compare_point_item(
        weight_SWCKM_30, lambda_, Ny, Nz, d, math.radians(phi0_30), math.radians(theta0_30), NA, NE, eps)
    pattern_dbw_WCKM_30, theta_WCKM_30, phi_WCKM_30, pattern_dbw_theta_WCKM_30, pattern_dbw_phi_WCKM_30 = paper_img_compare_point_item(
        weight_WCKM_30, lambda_, Ny, Nz, d, math.radians(phi0_30), math.radians(theta0_30), NA, NE, eps)
    pattern_dbw_NM_30, theta_NM_30, phi_NM_30, pattern_dbw_theta_NM_30, pattern_dbw_phi_NM_30 = paper_img_compare_point_item(
        weight_NM_30, lambda_, Ny, Nz, d, math.radians(phi0_30), math.radians(theta0_30), NA, NE, eps)
    pattern_dbw_NN_30, theta_NN_30, phi_NN_30, pattern_dbw_theta_NN_30, pattern_dbw_phi_NN_30 = paper_img_compare_point_item(
        weight_NN_30, lambda_, Ny, Nz, d, math.radians(phi0_30), math.radians(theta0_30), NA, NE, eps)
    #
    plt.figure()
    #
    plt.plot(theta_NN_0 * 180 / np.pi, pattern_dbw_theta_NN_0, label='UPM (0°)', color='#1f77b4', linewidth=2.5)
    plt.plot(theta_NM_0 * 180 / np.pi, pattern_dbw_theta_NM_0, label='NM (0°)', color='#2ca02c', linewidth=2.5)
    plt.plot(theta_CKM_0 * 180 / np.pi, pattern_dbw_theta_CKM_0, label='KCM (0°)', color='#ff7f0e', linewidth=2.5)
    plt.plot(theta_WCKM_0 * 180 / np.pi, pattern_dbw_theta_WCKM_0, label='WKCM (0°)', color='#9467bd', linewidth=2.5)
    plt.plot(theta_SWCKM_0 * 180 / np.pi, pattern_dbw_theta_SWCKM_0, label='SWKCM (0°)', color='#d62728', linewidth=2.5)
    plt.plot(theta_NM_30 * 180 / np.pi, pattern_dbw_theta_NM_30, label='NM (30°)', color='#2ca02c', linestyle='--',
             linewidth=3)
    plt.plot(theta_NN_30 * 180 / np.pi, pattern_dbw_theta_NN_30, label='UPM (30°)', color='#1f77b4', linestyle='--',
             linewidth=3)
    plt.plot(theta_CKM_30 * 180 / np.pi, pattern_dbw_theta_CKM_30, label='KCM (30°)', color='#ff7f0e', linestyle='--',
             linewidth=3)
    plt.plot(theta_WCKM_30 * 180 / np.pi, pattern_dbw_theta_WCKM_30, label='WKCM (30°)', color='#9467bd',
             linestyle='--', linewidth=3)
    plt.plot(theta_SWCKM_30 * 180 / np.pi, pattern_dbw_theta_SWCKM_30, label='SWKCM (30°)', color='#d62728',
             linestyle='--', linewidth=3)
    #
    plt.grid()
    plt.legend(ncol=2, loc='upper right', fontsize=16)  # 图例位于上方右侧
    plt.xlabel('theta (degree)', fontsize=18)
    plt.ylabel('normalized pattern (dB)', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
    #
    #
    plt.plot(phi_CKM_0 * 180 / np.pi, pattern_dbw_phi_CKM_0, label='KCM', color='#ff7f0e')
    plt.plot(phi_SWCKM_0 * 180 / np.pi, pattern_dbw_phi_SWCKM_0, label='SWKCM', color='#d62728')
    plt.plot(phi_WCKM_0 * 180 / np.pi, pattern_dbw_phi_WCKM_0, label='WKCM', color='#9467bd')
    plt.plot(phi_NM_0 * 180 / np.pi, pattern_dbw_phi_NM_0, label='NM', color='#2ca02c')
    plt.plot(phi_NN_0 * 180 / np.pi, pattern_dbw_phi_NN_0, label='UPM', color='#1f77b4')
    #
    # plt.plot(phi_CKM_15 * 180 / np.pi, pattern_dbw_phi_CKM_15, label='KCM', color='#ff7f0e', linestyle='--')
    # plt.plot(phi_SWCKM_15 * 180 / np.pi, pattern_dbw_phi_SWCKM_15, label='SWKCM', color='#d62728', linestyle='--')
    # plt.plot(phi_WCKM_15 * 180 / np.pi, pattern_dbw_phi_WCKM_15, label='WKCM', color='#9467bd', linestyle='--')
    # plt.plot(phi_NM_15 * 180 / np.pi, pattern_dbw_phi_NM_15, label='NM', color='#2ca02c', linestyle='--')
    # plt.plot(phi_NN_15 * 180 / np.pi, pattern_dbw_phi_NN_15, label='UPM', color='#1f77b4', linestyle='--')
    #
    plt.plot(phi_CKM_30 * 180 / np.pi, pattern_dbw_phi_CKM_30, label='KCM', color='#ff7f0e', linestyle=':')
    plt.plot(phi_SWCKM_30 * 180 / np.pi, pattern_dbw_phi_SWCKM_30, label='SWKCM', color='#d62728', linestyle=':')
    plt.plot(phi_WCKM_30 * 180 / np.pi, pattern_dbw_phi_WCKM_30, label='WKCM', color='#9467bd', linestyle=':')
    plt.plot(phi_NM_30 * 180 / np.pi, pattern_dbw_phi_NM_30, label='NM', color='#2ca02c', linestyle=':')
    plt.plot(phi_NN_30 * 180 / np.pi, pattern_dbw_phi_NN_30, label='UPM', color='#1f77b4', linestyle=':')
    #
    plt.grid()
    plt.legend()
    plt.xlabel('phi (degree)')
    plt.ylabel('normalized pattern (dB)')
    # plt.title('Chebyshev Plane Array (15x15 elements, 15 subarray)')
    plt.show()


def drawWeight(data):
    plt.figure()
    plt.imshow(data)
    plt.axis('off')  # 这将隐藏x轴和y轴
    plt.show()


def draw_af_theta2(lambda_, d, Ny, Nz, theta0, phi0, weight, NA, NE, eps, path_img):
    # 计算方向图
    pattern_dbw, theta, phi = arr_chebyshev_plane.compute_array_pattern(lambda_, d, Ny, Nz, math.radians(theta0), math.radians(phi0), weight, NA, NE, eps)
    # pattern_dbw, Theta, Phi, theta, phi = arr_chebyshev_plane_2.compute_array_pattern(lambda_, d, Ny, Nz, math.radians(theta0), math.radians(phi0), np.array(weight), NA, NE, eps)
    # 设置下限
    # pattern_dbw[pattern_dbw < -50] = -50 + np.random.uniform(-1, 1, np.sum(pattern_dbw < -50))
    # pattern_dbw[pattern_dbw < -80] = -80 + np.random.uniform(-1, 1, np.sum(pattern_dbw < -80))
    # 绘制方向图 -- xyz方向图
    # 转换为直角坐标系
    th, ph = np.meshgrid(theta, phi)
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = np.cos(th)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x, y, pattern_dbw, shading='auto', cmap='viridis')
    plt.axis('equal')
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()
    # plt.savefig(path_img, dpi=200, bbox_inches='tight', pad_inches=0)
    # plt.close(fig=None)  # 关闭当前图像窗口，如果fig=None，则默认关闭最近打开的图像窗口
    # plt.savefig(path_img, bbox_inches='tight')
    # plt.close()
    # 绘制方向图
    phi_deg = phi * 180 / np.pi
    theta_deg = theta * 180 / np.pi
    Phi, Theta = np.meshgrid(phi_deg, theta_deg)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Phi, Theta, pattern_dbw.T, cmap='viridis')  # 注意 pattern_dbw 需转置
    ax.set_xlabel('Phi (Degrees)')
    ax.set_ylabel('Theta (Degrees)')
    ax.set_title('Dolph-Chebyshev Planar Array Radiation Pattern')
    plt.show()
    # 绘制方向图
    plt.figure()
    plt.pcolormesh(theta * 180 / np.pi, phi * 180 / np.pi, pattern_dbw, shading='auto')
    plt.colorbar()
    plt.xlabel('Phi (Degrees)')
    plt.ylabel('Theta (Degrees)')
    plt.title('Dolph-Chebyshev Planar Array Radiation Pattern')
    plt.tight_layout()
    plt.show()
    # plt.savefig(path_img, bbox_inches='tight')
    # plt.close()




if __name__ == '__main__':
    lambda_ = 1  # 波长
    d = 0.5  # 阵元间隔
    Ny = 40  # 方位阵元个数
    Nz = 40  # 俯仰阵元个数
    eps = 0.0001  # 底电平
    NA = 360  # 方位角度采样
    NE = 360  # 俯仰角度采样
    SLL = -60
    k_kmeans = 9  # 子阵数量
    #
    #
    # 波束指向角
    phi0_0 = 0  # 方位指向
    theta0_0 = 0  # 俯仰指向
    phi0_15 = 0  # 方位指向
    theta0_15 = 15  # 俯仰指向
    phi0_30 = 0  # 方位指向
    theta0_30 = 30  # 俯仰指向

    # 读取CSV文件
    # file_path_CKM = "./files/15x15-10/cluster_kmeans_chebyshev_plane_spare_arr_w_20240307.csv"
    # file_path_SWCKM = "./files/15x15-10/cluster_kmeans_weighted_chebyshev_plane_spare_arr_w_20240307.csv"
    # file_path_NM = "./files/15x15-10/subarrayNM_15x15_10_2024-06-13.csv"
    # # file_path_GANM = "./files/15x15-10/subarrayGA-NM_15x15_10_2024-06-13.csv"
    # file_path_GANM = "./files/15x15-10/subarrayGA-NM_15x15_10_30-3-3_2024-06-13.csv"
    #
    # file_path_CKM = "./files/15x15-10/6-14/worst/cluster_kmeans_chebyshev_plane_worst_arr_w_20240614_2.csv"
    # file_path_SWCKM = "./files/15x15-10/6-14/worst/cluster_kmeans_weighted_chebyshev_plane_worst_arr_w_20240614_2.csv"
    # file_path_NM = "./files/15x15-10/6-14/subarrayNM_15x15_10_2024-06-13.csv"
    # file_path_GANM = "./files/15x15-10/6-14/worst/subarrayGA_15x15_10_30-3-3_worst_2024-06-13.csv"
    #
    # file_path_CKM = "./files/15x15-10/6-14/best/cluster_kmeans_chebyshev_plane_spare_arr_w_20240614_2.csv"
    # file_path_SWCKM = "./files/15x15-10/6-14/best/cluster_kmeans_weighted_chebyshev_plane_spare_arr_w_20240614_2.csv"
    # file_path_NM = "./files/15x15-10/6-14/subarrayNM_15x15_10_2024-06-13.csv"
    # file_path_GANM = "./files/15x15-10/6-14/best/subarrayGA_15x15_10_30-3-3_2024-06-13.csv"
    #
    # file_path_CKM = "./files/40x40-9-sll60/best_arr_w_kcm_001.csv"
    # file_path_SWCKM = "./files/40x40-9-sll60/best_arr_w_swkcm_001.csv"
    # file_path_WCKM = "./files/40x40-9-sll60/best_arr_w_[6]_wkcm_004.csv"
    # file_path_NM = "./files/40x40-9-sll60/subarrayNM_40x40-9_60_2024-06-27.csv"
    # file_path_NN = "./files/40x40-9-sll60/subarrayNN_40x40-9_60_2024-06-28.csv"
    # file_path_GANM = "./files/40x40-9-sll60/"
    #
    #
    file_path_CKM_0 = "./files/40x40-9-sll60/best_arr_w_kcm_001.csv"
    file_path_SWCKM_0 = "./files/40x40-9-sll60/best_arr_w_swkcm_001.csv"
    file_path_WCKM_0 = "./files/40x40-9-sll60/best_arr_w_[6]_wkcm_004.csv"
    file_path_NM_0 = "./files/40x40-9-sll60/subarrayNM_40x40-9_60_2024-06-27.csv"
    file_path_NN_0 = "./files/40x40-9-sll60/subarrayNN_40x40-9_60_2024-06-28.csv"
    #
    file_path_CKM_15 = "./files/40x40-9-sll60/best_arr_w_kcm_001.csv"
    file_path_SWCKM_15 = "./files/40x40-9-sll60/best_arr_w_swkcm_001.csv"
    file_path_WCKM_15 = "./files/40x40-9-sll60/best_arr_w_[6]_wkcm_004.csv"
    file_path_NM_15 = "./files/40x40-9-sll60/subarrayNM_40x40-9_60_2024-06-27.csv"
    file_path_NN_15 = "./files/40x40-9-sll60/subarrayNN_40x40-9_60_2024-06-28.csv"
    #
    file_path_CKM_30 = "./files/40x40-9-sll60/best_arr_w_kcm_001.csv"
    file_path_SWCKM_30 = "./files/40x40-9-sll60/best_arr_w_swkcm_001.csv"
    file_path_WCKM_30 = "./files/40x40-9-sll60/best_arr_w_[6]_wkcm_004.csv"
    file_path_NM_30 = "./files/40x40-9-sll60/subarrayNM_40x40-9_60_2024-06-27.csv"
    file_path_NN_30 = "./files/40x40-9-sll60/subarrayNN_40x40-9_60_2024-06-28.csv"
    #
    #
    weight_CKM_0 = read_csv_to_2d_array_with_numbers(file_path_CKM_0)
    weight_SWCKM_0 = read_csv_to_2d_array_with_numbers(file_path_SWCKM_0)
    weight_WCKM_0 = read_csv_to_2d_array_with_numbers(file_path_WCKM_0)
    weight_NM_0 = read_csv_to_2d_array_with_numbers(file_path_NM_0)
    weight_NN_0 = read_csv_to_2d_array_with_numbers(file_path_NN_0)
    weight_NN_0 = np.array(weight_NN_0)[1:41, 1:41]
    #
    weight_CKM_15 = read_csv_to_2d_array_with_numbers(file_path_CKM_15)
    weight_SWCKM_15 = read_csv_to_2d_array_with_numbers(file_path_SWCKM_15)
    weight_WCKM_15 = read_csv_to_2d_array_with_numbers(file_path_WCKM_15)
    weight_NM_15 = read_csv_to_2d_array_with_numbers(file_path_NM_15)
    weight_NN_15 = read_csv_to_2d_array_with_numbers(file_path_NN_15)
    weight_NN_15 = np.array(weight_NN_15)[1:41, 1:41]
    #
    weight_CKM_30 = read_csv_to_2d_array_with_numbers(file_path_CKM_30)
    weight_SWCKM_30 = read_csv_to_2d_array_with_numbers(file_path_SWCKM_30)
    weight_WCKM_30 = read_csv_to_2d_array_with_numbers(file_path_WCKM_30)
    weight_NM_30 = read_csv_to_2d_array_with_numbers(file_path_NM_30)
    weight_NN_30 = read_csv_to_2d_array_with_numbers(file_path_NN_30)
    weight_NN_30 = np.array(weight_NN_30)[1:41, 1:41]
    # weight_GANM_pre = read_csv_to_2d_array_with_numbers(file_path_GANM)
    # weight_GANM = average_values_by_indices(weight_GANM_pre)
    #
    #
    # 论文图 -- 单图子阵划分结果
    # drawWeight(weight_CKM)
    # drawWeight(weight_SWCKM)
    # drawWeight(weight_WCKM)
    # drawWeight(weight_NM)
    # drawWeight(weight_NN)
    # drawWeight(weight_GANM)
    #
    # 论文图 -- 比较
    # paper_img_compare_2(weight_CKM, weight_SWCKM, weight_NM, weight_NN, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps)
    # paper_img_compare_3(weight_CKM, weight_SWCKM, weight_WCKM, weight_NM, weight_NN, lambda_, Ny, Nz, d, phi0, theta0, NA, NE, eps)
    #
    # 论文图 -- 比较（单一）
    labels = ['UPM (0°)', 'NM (0°)', 'KCM (0°)', 'WKCM (0°)', 'SWKCM (0°)']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
    linewidths = [2.5, 2.5, 2.5, 2.5, 2.5]
    linestyles = ['-', '-', '-', '-', '-']
    #
    # paper_img_compare_point(weight_CKM_0, weight_SWCKM_0, weight_WCKM_0, weight_NM_0, weight_NN_0, lambda_, Ny, Nz, d,
    #                         math.radians(phi0_0), math.radians(theta0_0), NA, NE, eps,
    #                         ['UPM (0°)', 'NM (0°)', 'KCM (0°)', 'WKCM (0°)', 'SWKCM (0°)'],
    #                         ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728'],
    #                         [4.5, 4.5, 4.5, 4.5, 4.5],
    #                         ['-', '-', '-', '-', '-'])
    # paper_img_compare_point(weight_CKM_15, weight_SWCKM_15, weight_WCKM_15, weight_NM_15, weight_NN_15, lambda_, Ny, Nz, d,
    #                         math.radians(phi0_15), math.radians(theta0_15), NA, NE, eps,
    #                         ['UPM (15°)', 'NM (15°)', 'KCM (15°)', 'WKCM (15°)', 'SWKCM (15°)'],
    #                         ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728'],
    #                         [3, 3, 3, 3, 3],
    #                         [':', ':', ':', ':', ':'])
    # paper_img_compare_point(weight_CKM_30, weight_SWCKM_30, weight_WCKM_30, weight_NM_30, weight_NN_30, lambda_, Ny, Nz, d,
    #                         math.radians(phi0_30), math.radians(theta0_30), NA, NE, eps,
    #                         ['UPM (30°)', 'NM (30°)', 'KCM (30°)', 'WKCM (30°)', 'SWKCM (30°)'],
    #                         ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728'],
    #                         [5, 5, 5, 5, 5],
    #                         ['--', '--', '--', '--', '--'])
    #
    # 论文图 -- 比较（全部）
    # paper_img_compare_point_3(weight_CKM_0, weight_SWCKM_0, weight_WCKM_0, weight_NM_0, weight_NN_0, phi0_0, theta0_0,
    #                           weight_CKM_15, weight_SWCKM_15, weight_WCKM_15, weight_NM_15, weight_NN_15, phi0_15, theta0_15,
    #                           weight_CKM_30, weight_SWCKM_30, weight_WCKM_30, weight_NM_30, weight_NN_30, phi0_30, theta0_30,
    #                           lambda_, Ny, Nz, d, NA, NE, eps)
    #
    # 论文图 -- 俯视方向图
    # draw_af_theta2(lambda_, d, Ny, Nz, theta0_0, phi0_0, weight_NN_0, NA, NE, eps, "./files/subarray/40x40-9/pattern-40x40-9-0-NN.jpg")
    # draw_af_theta2(lambda_, d, Ny, Nz, theta0_0, phi0_0, weight_NM_0, NA, NE, eps, "./files/subarray/40x40-9/pattern-40x40-9-0-NM.jpg")
    # draw_af_theta2(lambda_, d, Ny, Nz, theta0_0, phi0_0, weight_CKM_0, NA, NE, eps, "./files/subarray/40x40-9/pattern-40x40-9-0-KCM.jpg")
    # draw_af_theta2(lambda_, d, Ny, Nz, theta0_0, phi0_0, weight_WCKM_0, NA, NE, eps, "./files/subarray/40x40-9/pattern-40x40-9-0-WKCM.jpg")
    # draw_af_theta2(lambda_, d, Ny, Nz, theta0_0, phi0_0, weight_SWCKM_0, NA, NE, eps, "./files/subarray/40x40-9/pattern-40x40-9-0-SWKCM.jpg")
    # #
    # draw_af_theta2(lambda_, d, Ny, Nz, theta0_30, phi0_30, weight_NN_30, NA, NE, eps, "./files/subarray/40x40-9/pattern-40x40-9-30-NN.jpg")
    # draw_af_theta2(lambda_, d, Ny, Nz, theta0_30, phi0_30, weight_NM_30, NA, NE, eps, "./files/subarray/40x40-9/pattern-40x40-9-30-NM.jpg")
    # draw_af_theta2(lambda_, d, Ny, Nz, theta0_30, phi0_30, weight_CKM_30, NA, NE, eps, "./files/subarray/40x40-9/pattern-40x40-9-30-KCM.jpg")
    # draw_af_theta2(lambda_, d, Ny, Nz, theta0_30, phi0_30, weight_WCKM_30, NA, NE, eps, "./files/subarray/40x40-9/pattern-40x40-9-30-WKCM.jpg")
    draw_af_theta2(lambda_, d, Ny, Nz, theta0_30, phi0_30, weight_SWCKM_30, NA, NE, eps, "./files/subarray/40x40-9/pattern-40x40-9-30-SWKCM.jpg")