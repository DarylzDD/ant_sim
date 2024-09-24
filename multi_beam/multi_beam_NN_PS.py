import logging
import random
import numpy as np
import matplotlib.pyplot as plt

from util.util_log import setup_logging
from util.util_csv import save_csv
from util.util_image import save_img, save_img_xyz, draw_img
from util.util_ris_pattern import point_2_phi_pattern, phase_2_pattern, nRow, eps, phase_2_pattern_xyz, phase_2_bit
from util.util_analysis_plane import get_peaks, get_peak_3rd
from util.util_statistics import calculate_statistics

from multi_beam_NN import nn_beam_2_ud, nn_beam_4, nn_beam_8_cub
from multi_beam_PS import psm_beam_2, psm_beam_4, psm_beam_n


def fill_array(M, N):
    # 创建一个 MxM 的零数组
    array = np.zeros((M, M), dtype=int)
    # 填充数组
    for i in range(M):
        for j in range(M):
            array[i, j] = (i * M + j) % N
    return array


def fill_array_with_arrays(M, arrays):
    N = len(arrays)
    # 创建一个 MxM 的零数组
    array = np.zeros((M, M), dtype=int)
    # 填充数组
    for i in range(M):
        for j in range(M):
            n = (i * M + j) % N
            n_arrays = arrays[n]
            array[i, j] = n_arrays[i, j]
    return array


def get_ps_block(buffer_percent, side_len, side_mid, side_start, side_end):
    buffer_height = int(side_len * buffer_percent / 100)
    # rows
    start = side_mid - buffer_height
    end = side_mid + buffer_height
    # 确保索引在矩阵范围内
    start = max(side_start, start)
    end = min(side_end, end)
    return start, end


# ============================================= ris的"1"随机取150°~200° ====================================
# 统计N次1bit中"1"随机取角度的PSLL, 阵列, 方向图
def multi_beam_2_rand(iteration, phaseBit_mix_ud, phaseBit_mix_lr):
    list_psll_ud = []
    best_psll_ud = 0
    best_phaseBit_ud = None
    best_phaseBitDeg_ud = None
    best_patternBit_ud = None
    worst_psll_ud = -200
    worst_phaseBit_ud = None
    worst_phaseBitDeg_ud = None
    worst_patternBit_ud = None
    #
    list_psll_lr = []
    best_psll_lr = 0
    best_phaseBit_lr = None
    best_phaseBitDeg_lr = None
    best_patternBit_lr = None
    worst_psll_lr = -200
    worst_phaseBit_lr = None
    worst_phaseBitDeg_lr = None
    worst_patternBit_lr = None
    #
    for i in range(iteration):
        # phaseBit_mix_ud_rand 和 phaseBit_mix_lr_rand: 连续相位转1bit相位, 1时随机取150°~200°
        random_degrees = np.random.randint(150, 201, size=phaseBit_mix_ud.shape)
        phaseBit_mix_ud_rand_deg = phaseBit_mix_ud * random_degrees
        phaseBit_mix_ud_rand = np.deg2rad(phaseBit_mix_ud_rand_deg)
        random_degrees = np.random.randint(150, 201, size=phaseBit_mix_lr.shape)
        phaseBit_mix_lr_rand_deg = phaseBit_mix_lr * random_degrees
        phaseBit_mix_lr_rand = np.deg2rad(phaseBit_mix_lr_rand_deg)
        # 计算方向图 pattern 和 pattern_dbw
        patternBit_mix_ud_rand = phase_2_pattern(phaseBit_mix_ud_rand)
        patternBit_mix_ud_rand_dbw = 20 * np.log10(
            np.abs(patternBit_mix_ud_rand) / np.max(np.max(np.abs(patternBit_mix_ud_rand))) + eps)
        patternBit_mix_lr_rand = phase_2_pattern(phaseBit_mix_lr_rand)
        patternBit_mix_lr_rand_dbw = 20 * np.log10(
            np.abs(patternBit_mix_lr_rand) / np.max(np.max(np.abs(patternBit_mix_lr_rand))) + eps)
        # 计算PSLL
        peaks_ud = get_peaks(patternBit_mix_ud_rand_dbw)
        peaks_ud_3rd = get_peak_3rd(peaks_ud)
        peaks_lr = get_peaks(patternBit_mix_lr_rand_dbw)
        peaks_lr_3rd = get_peak_3rd(peaks_lr)
        # 保存每次的PSLL, 最好和最差的阵列和方向图
        if peaks_ud_3rd is not None:
            peaks_ud_3rd_val = peaks_ud_3rd[0]
            list_psll_ud.append(peaks_ud_3rd_val)
            if peaks_ud_3rd_val > worst_psll_ud:
                worst_psll_ud = peaks_ud_3rd_val
                worst_phaseBit_ud = phaseBit_mix_ud_rand
                worst_phaseBitDeg_ud = phaseBit_mix_ud_rand_deg
                worst_patternBit_ud = patternBit_mix_ud_rand
            if peaks_ud_3rd_val < best_psll_ud:
                best_psll_ud = peaks_ud_3rd_val
                best_phaseBit_ud = phaseBit_mix_ud_rand
                best_phaseBitDeg_ud = phaseBit_mix_ud_rand_deg
                best_patternBit_ud = patternBit_mix_ud_rand
        if peaks_lr_3rd is not None:
            peaks_lr_3rd_val = peaks_lr_3rd[0]
            list_psll_lr.append(peaks_lr_3rd_val)
            if peaks_lr_3rd_val > worst_psll_lr:
                worst_psll_lr = peaks_lr_3rd_val
                worst_phaseBit_lr = phaseBit_mix_lr_rand
                worst_phaseBitDeg_lr = phaseBit_mix_lr_rand_deg
                worst_patternBit_lr = patternBit_mix_lr_rand
            if peaks_lr_3rd_val < best_psll_lr:
                best_psll_lr = peaks_lr_3rd_val
                best_phaseBit_lr = phaseBit_mix_lr_rand
                best_phaseBitDeg_lr = phaseBit_mix_lr_rand_deg
                best_patternBit_lr = patternBit_mix_lr_rand
        logger.info(
            "i=%d, peaks_ud_3rd=%s, peaks_lr_3rd=%s, best_psll_ud=%f, best_psll_lr=%f, worst_psll_ud=%f, worst_psll_lr=%f" % (
                i, peaks_ud_3rd, peaks_lr_3rd, best_psll_ud, best_psll_lr, worst_psll_ud, worst_psll_lr))
    return list_psll_ud, best_psll_ud, best_phaseBit_ud, best_phaseBitDeg_ud, best_patternBit_ud, \
           worst_psll_ud, worst_phaseBit_ud, worst_phaseBitDeg_ud, worst_patternBit_ud, \
           list_psll_lr, best_psll_lr, best_phaseBit_lr, best_phaseBitDeg_lr, best_patternBit_lr, \
           worst_psll_lr, worst_phaseBit_lr, worst_phaseBitDeg_lr, worst_patternBit_lr


# ============================================= 测试函数 ====================================
def test_split_array():
    np.set_printoptions(threshold=np.inf)  # 设置为无限制
    # 示例
    M1, N1 = 4, 4
    array1 = fill_array(M1, N1)
    print("M=4, N=4:")
    print(array1)

    M2, N2 = 4, 3
    array2 = fill_array(M2, N2)
    print("\nM=4, N=3:")
    print(array2)

    arrays = []
    for i in range(8):
        arrays.append(np.full((16, 16), i))
    array = fill_array_with_arrays(16, arrays)
    print("\narray:")
    print(array)


def test_phaseBit_mix_8():
    rows, cols = 64, 64
    phaseBit1 = np.full((rows, cols), 1)
    phaseBit2 = np.full((rows, cols), 2)
    phaseBit3 = np.full((rows, cols), 3)
    phaseBit4 = np.full((rows, cols), 4)
    phaseBit5 = np.full((rows, cols), 5)
    phaseBit6 = np.full((rows, cols), 6)
    phaseBit7 = np.full((rows, cols), 7)
    phaseBit8 = np.full((rows, cols), 8)
    phaseBit_mix = np.zeros((rows, cols))
    # 分块方式: 1.cub
    # # 竖切
    # # phaseBit_mix[:rows // 2, cols // 2:] = phaseBit1[:rows // 2, cols // 2:]  # 右上半部分
    # # phaseBit_mix[rows // 2:, cols // 2:] = phaseBit3[rows // 2:, cols // 2:]  # 右下半部分
    # # phaseBit_mix[rows // 2:, :cols // 2] = phaseBit5[rows // 2:, :cols // 2]  # 左下半部分
    # # phaseBit_mix[:rows // 2, :cols // 2] = phaseBit7[:rows // 2, :cols // 2]  # 左上半部分
    # # phaseBit_mix[:rows // 2, cols // 2:cols // 2 + cols // 4] = phaseBit2[:rows // 2, cols // 2:cols // 2 + cols // 4]  # 右上中部分（左半部分）
    # # phaseBit_mix[rows // 2:, cols // 2:cols // 2 + cols // 4] = phaseBit4[rows // 2:, cols // 2:cols // 2 + cols // 4]  # 右下中部分（左半部分）
    # # phaseBit_mix[rows // 2:, :cols // 4] = phaseBit6[rows // 2:, :cols // 4]  # 左下中部分（左半部分）
    # # phaseBit_mix[:rows // 2, :cols // 4] = phaseBit8[:rows // 2, :cols // 4]  # 左上中部分（左半部分）
    # # 横切
    # phaseBit_mix[:rows // 2, cols // 2:] = phaseBit8[:rows // 2, cols // 2:]  # 右上半部分
    # phaseBit_mix[rows // 2:, cols // 2:] = phaseBit7[rows // 2:, cols // 2:]  # 右下半部分
    # phaseBit_mix[rows // 2:, :cols // 2] = phaseBit4[rows // 2:, :cols // 2]  # 左下半部分
    # phaseBit_mix[:rows // 2, :cols // 2] = phaseBit2[:rows // 2, :cols // 2]  # 左上半部分
    # phaseBit_mix[rows // 4:rows // 2, cols // 2:] = phaseBit1[rows // 4:rows // 2, cols // 2:]  # 右上中部分（下半部分）
    # phaseBit_mix[rows // 2 + rows // 4:, cols // 2:] = phaseBit6[rows // 2 + rows // 4:, cols // 2:]  # 右下中部分（下半部分）
    # phaseBit_mix[rows // 2 + rows // 4:, :cols // 2] = phaseBit5[rows // 2 + rows // 4:, :cols // 2]  # 左下中部分（左半部分）
    # phaseBit_mix[rows // 4:rows // 2, :cols // 2] = phaseBit3[rows // 4:rows // 2, :cols // 2]  # 左上中部分（左半部分）
    #
    # 分块方式: 2.ang
    # # 左上部分（对角线从左上到右下）
    # phaseBit_mix[:rows // 2, :cols // 2] = np.where(np.arange(rows // 2)[:, None] < np.arange(cols // 2),
    #                                                 phaseBit1[rows // 2:, cols // 2:],
    #                                                 phaseBit8[rows // 2:, cols // 2:])
    # # 右上部分（对角线从右上到左下）
    # phaseBit_mix[:rows // 2, cols // 2:] = np.where(np.arange(rows // 2)[:, None] < (cols // 2) - np.arange(rows // 2),
    #                                                 phaseBit2[:rows // 2, :cols // 2],
    #                                                 phaseBit3[:rows // 2, :cols // 2])
    # # 右下部分（对角线从左上到右下）
    # phaseBit_mix[rows // 2:, cols // 2:] = np.where(np.arange(rows // 2)[:, None] < np.arange(cols // 2),
    #                                                 phaseBit4[rows // 2:, cols // 2:],
    #                                                 phaseBit5[rows // 2:, cols // 2:])
    # # 左下部分（对角线从右上到左下）
    # phaseBit_mix[rows // 2:, :cols // 2] = np.where(np.arange(rows // 2)[:, None] < (cols // 2) - np.arange(rows // 2),
    #                                                 phaseBit7[rows // 2:, :cols // 2],
    #                                                 phaseBit6[rows // 2:, :cols // 2])
    #
    # 分块方式: 3.散列
    phaseBit_list = [phaseBit1, phaseBit2, phaseBit3, phaseBit4, phaseBit5, phaseBit6, phaseBit7, phaseBit8]
    phaseBit_mix = fill_array_with_arrays(nRow, phaseBit_list)
    #
    draw_img(phaseBit_mix)


def test_phaseBit_mix_2():
    rows, cols = 64, 64
    buffer_percent = 15
    phaseBit1 = np.full((rows, cols), 1)
    phaseBit2 = np.full((rows, cols), 2)
    phaseBit_mix = np.zeros((rows, cols))
    # 1.NN填充
    # # 上下填充
    phaseBit_mix[:rows // 2, :] = phaseBit1[:rows // 2, :]
    phaseBit_mix[rows // 2:, :] = phaseBit2[rows // 2:, :]
    # 2.PS计算过度区间
    # # 以phaseBit1和phaseBit2边界为中线，高度为buffer_percent，宽度为cols的phaseBit_mix矩形区间里，
    # # phaseBit_mix值变为该位置phaseBit1+phaseBit2/2
    buffer_height = int(rows * buffer_percent / 100)
    start_row = rows // 2 - buffer_height
    end_row = rows // 2 + buffer_height
    # 确保索引在矩阵范围内
    start_row = max(0, start_row)
    end_row = min(rows, end_row)
    for i in range(start_row, end_row):
        phaseBit_mix[i, :] = (phaseBit1[i, :] + phaseBit2[i, :]) / 2
    #
    draw_img(phaseBit_mix)


def test_phaseBit_mix_16():
    rows, cols = 64, 64
    phaseBit1 = np.full((rows, cols), 1)
    phaseBit2 = np.full((rows, cols), 2)
    phaseBit3 = np.full((rows, cols), 3)
    phaseBit4 = np.full((rows, cols), 4)
    phaseBit5 = np.full((rows, cols), 5)
    phaseBit6 = np.full((rows, cols), 6)
    phaseBit7 = np.full((rows, cols), 7)
    phaseBit8 = np.full((rows, cols), 8)
    phaseBit9 = np.full((rows, cols), 9)
    phaseBit10 = np.full((rows, cols), 10)
    phaseBit11 = np.full((rows, cols), 11)
    phaseBit12 = np.full((rows, cols), 12)
    phaseBit13 = np.full((rows, cols), 13)
    phaseBit14 = np.full((rows, cols), 14)
    phaseBit15 = np.full((rows, cols), 15)
    phaseBit16 = np.full((rows, cols), 16)
    phaseBit_mix = np.zeros((rows, cols))
    # 填充 phase_mix 的16个部分
    phaseBit_mix[:rows // 4, :cols // 4] = phaseBit1[:rows // 4, :cols // 4]
    phaseBit_mix[:rows // 4, cols // 4:cols // 2] = phaseBit2[:rows // 4, cols // 4:cols // 2]
    phaseBit_mix[:rows // 4, cols // 2:3 * cols // 4] = phaseBit3[:rows // 4, cols // 2:3 * cols // 4]
    phaseBit_mix[:rows // 4, 3 * cols // 4:] = phaseBit4[:rows // 4, 3 * cols // 4:]

    phaseBit_mix[rows // 4:rows // 2, :cols // 4] = phaseBit5[rows // 4:rows // 2, :cols // 4]
    phaseBit_mix[rows // 4:rows // 2, cols // 4:cols // 2] = phaseBit6[rows // 4:rows // 2, cols // 4:cols // 2]
    phaseBit_mix[rows // 4:rows // 2, cols // 2:3 * cols // 4] = phaseBit7[rows // 4:rows // 2, cols // 2:3 * cols // 4]
    phaseBit_mix[rows // 4:rows // 2, 3 * cols // 4:] = phaseBit8[rows // 4:rows // 2, 3 * cols // 4:]

    phaseBit_mix[rows // 2:3 * rows // 4, :cols // 4] = phaseBit9[rows // 2:3 * rows // 4, :cols // 4]
    phaseBit_mix[rows // 2:3 * rows // 4, cols // 4:cols // 2] = phaseBit10[rows // 2:3 * rows // 4,
                                                                 cols // 4:cols // 2]
    phaseBit_mix[rows // 2:3 * rows // 4, cols // 2:3 * cols // 4] = phaseBit11[rows // 2:3 * rows // 4,
                                                                     cols // 2:3 * cols // 4]
    phaseBit_mix[rows // 2:3 * rows // 4, 3 * cols // 4:] = phaseBit12[rows // 2:3 * rows // 4, 3 * cols // 4:]

    phaseBit_mix[3 * rows // 4:, :cols // 4] = phaseBit13[3 * rows // 4:, :cols // 4]
    phaseBit_mix[3 * rows // 4:, cols // 4:cols // 2] = phaseBit14[3 * rows // 4:, cols // 4:cols // 2]
    phaseBit_mix[3 * rows // 4:, cols // 2:3 * cols // 4] = phaseBit15[3 * rows // 4:, cols // 2:3 * cols // 4]
    phaseBit_mix[3 * rows // 4:, 3 * cols // 4:] = phaseBit16[3 * rows // 4:, 3 * cols // 4:]
    #
    draw_img(phaseBit_mix)


# ============================================= 主函数 ====================================
# NN 核心算法 -- 双波束
def nn_ps_beam_2_ud(phase1, phase2, bit_num, buffer_percent):
    rows, cols = phase1.shape
    # 1.NN填充
    # # 上下填充
    phase_mix = nn_beam_2_ud(phase1, phase2)
    # 2.PS计算过度区间
    # PS 计算
    phase_mix_ps, phaseBit_mix_ps, phaseBitDeg_mix_ps = psm_beam_2(phase1, phase2, bit_num)
    # PS 填充
    # # 以phaseBit1和phaseBit2边界为中线，高度为buffer_percent，宽度为cols的phaseBit_mix矩形区间里，
    # # phaseBit_mix值变为该位置phaseBit1+phaseBit2/2
    start_row, end_row = get_ps_block(buffer_percent, rows, rows//2, 0, rows)
    for i in range(start_row, end_row):
        phase_mix[i, :] = phase_mix_ps[i, :]
    return phase_mix


# NN 核心算法 -- 四波束
def nn_ps_beam_4(phase1, phase2, phase3, phase4, bit_num, buffer_percent):
    rows, cols = phase1.shape
    # 1.NN填充
    # # 上下填充
    phase_mix = nn_beam_4(phase1, phase2, phase3, phase4)
    # 2.PS计算过度区间
    # PS 计算
    phase_mix_ps, phaseBit_mix_ps, phaseBitDeg_mix_ps = psm_beam_4(phase1, phase2, phase3, phase4, bit_num)
    # PS 填充
    # # 以phaseBit1和phaseBit2边界为中线，高度为buffer_percent，宽度为cols的phaseBit_mix矩形区间里，
    # # phaseBit_mix值变为该位置phaseBit1+phaseBit2/2
    start_row, end_row = get_ps_block(buffer_percent, rows, rows // 2, 0, rows)
    for i in range(start_row, end_row):
        phase_mix[i, :] = phase_mix_ps[i, :]
    start_col, end_col = get_ps_block(buffer_percent, cols, cols // 2, 0, cols)
    for i in range(start_col, end_col):
        phase_mix[:, i] = phase_mix_ps[:, i]
    return phase_mix


# NN 核心算法 -- 八波束 cub
def nn_ps_beam_8_cub(phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, bit_num, buffer_percent):
    rows, cols = phase1.shape
    # 1.NN填充
    # # 上下填充
    phase_mix = nn_beam_8_cub(phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8)
    # 2.PS计算过度区间
    # PS 计算
    phase_mix_ps, phaseBit_mix_ps, phaseBitDeg_mix_ps = psm_beam_n(
        [phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8], bit_num)
    # PS 填充
    # # 以phaseBit1和phaseBit2边界为中线，高度为buffer_percent，宽度为cols的phaseBit_mix矩形区间里，
    # # phaseBit_mix值变为该位置phaseBit1+phaseBit2/2
    start_row1, end_row1 = get_ps_block(buffer_percent, rows, rows // 2, 0, rows)
    for i in range(start_row1, end_row1):
        phase_mix[i, :] = phase_mix_ps[i, :]
    start_row2, end_row2 = get_ps_block(buffer_percent, rows, rows // 4, 0, rows // 2)
    for i in range(start_row2, end_row2):
        phase_mix[i, :] = phase_mix_ps[i, :]
    start_row3, end_row3 = get_ps_block(buffer_percent, rows, 3 * rows // 4, rows // 2, rows)
    for i in range(start_row3, end_row3):
        phase_mix[i, :] = phase_mix_ps[i, :]
    start_col, end_col = get_ps_block(buffer_percent, cols, cols // 2, 0, cols)
    for i in range(start_col, end_col):
        phase_mix[:, i] = phase_mix_ps[:, i]
    return phase_mix


# 几何分区法 -- 双波束
def main_multi_beam_2(theta1, phi1, theta2, phi2, path_pre, bit_num, buffer_percent):
    logger.info("main_multi_beam_2: bit_num=%d, path_pre=%s, " % (bit_num, path_pre))
    logger.info("main_multi_beam_2: theta1=%d, phi1=%d, theta2=%d, phi2=%d, " % (theta1, phi1, theta2, phi2))
    # 目前只支持2bit
    if bit_num > 2:
        logger.error("main_multi_beam_2: bit_num bigger than 2.")
        return
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1, bit_num)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2, bit_num)
    # 确保 phase1 和 phase2 具有相同的形状
    assert phaseBit1.shape == phaseBit2.shape, "phase1 和 phase2 必须具有相同的形状"
    # NN-PS
    phase_mix = nn_ps_beam_2_ud(phase1, phase2, bit_num, buffer_percent)
    # 相位转换 X bit
    phaseBit_mix, phaseBitDeg_mix = phase_2_bit(phase_mix, bit_num)
    # 计算phase_mix的方向图
    phaseBit_mix = np.deg2rad(phaseBitDeg_mix)
    patternBit_mix = phase_2_pattern(phaseBit_mix)
    #
    # 保存结果
    logger.info("save NN multi-beam 2 result...")
    patternBit_mix_xyz, x, y, z = phase_2_pattern_xyz(phaseBit_mix)
    # 保存图片
    save_img(path_pre + "phase1.jpg", phase1)
    save_img(path_pre + "phase2.jpg", phase2)
    save_img(path_pre + "phaseBit1.jpg", phaseBit1)
    save_img(path_pre + "phaseBit2.jpg", phaseBit2)
    save_img(path_pre + "pattern1.jpg", pattern1)
    save_img(path_pre + "pattern2.jpg", pattern2)
    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_mix)         # 几何分区法 -- 结果码阵
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)     # 几何分区法 -- 结果码阵方向图
    save_img_xyz(path_pre + "patternBit_mix_xyz.jpg", np.abs(patternBit_mix_xyz), x, y)
    # 保存相位结果
    save_csv(phase1, path_pre + "phase1.csv")
    save_csv(phase2, path_pre + "phase2.csv")
    save_csv(phaseBit1, path_pre + "phaseBit1.csv")
    save_csv(phaseBit2, path_pre + "phaseBit2.csv")
    save_csv(phaseBit_mix, path_pre + "phaseBit_mix.csv")
    # 保存分区结果
    rows, cols = phaseBit1.shape
    phase_cub_1 = np.full((rows, cols), 0)
    phase_cub_2 = np.full((rows, cols), 180)
    phase_cub = nn_ps_beam_2_ud(phase_cub_1, phase_cub_2, bit_num, buffer_percent)
    save_img(path_pre + "phase_cub.jpg", phase_cub)


# 几何分区法 -- 四波束
def main_multi_beam_4(theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4, path_pre, bit_num, buffer_percent):
    logger.info("main_multi_beam_4: bit_num=%d, path_pre=%s, " % (bit_num, path_pre))
    logger.info("main_multi_beam_4: theta1=%d, phi1=%d, theta2=%d, phi2=%d, theta3=%d, phi3=%d, theta4=%d, phi4=%d"
                % (theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4))
    # 目前只支持2bit
    if bit_num > 2:
        logger.error("main_multi_beam_N: bit_num bigger than 2.")
        return
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1, bit_num)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2, bit_num)
    phase3, phaseBit3, pattern3 = point_2_phi_pattern(theta3, phi3, bit_num)
    phase4, phaseBit4, pattern4 = point_2_phi_pattern(theta4, phi4, bit_num)
    # 确保所有数组具有相同的形状
    assert phaseBit1.shape == phaseBit2.shape == phaseBit3.shape == phaseBit4.shape, "所有数组必须具有相同的形状"
    # NN - PS
    phase_mix = nn_ps_beam_4(phase1, phase2, phase3, phase4, bit_num, buffer_percent)
    # 相位转换 X bit
    phaseBit_mix, phaseBitDeg_mix = phase_2_bit(phase_mix, bit_num)
    # 计算phase_mix的方向图
    phaseBit_mix = np.deg2rad(phaseBitDeg_mix)
    # 计算phase_mix
    patternBit_mix = phase_2_pattern(phaseBit_mix)
    #
    # 保存结果
    logger.info("save NN multi-beam 4 result...")
    patternBit_mix_xyz, x, y, z = phase_2_pattern_xyz(phaseBit_mix)
    # 保存图片
    save_img(path_pre + "phase1.jpg", phase1)
    save_img(path_pre + "phase2.jpg", phase2)
    save_img(path_pre + "phase3.jpg", phase3)
    save_img(path_pre + "phase4.jpg", phase4)
    save_img(path_pre + "phaseBit1.jpg", phaseBit1)
    save_img(path_pre + "phaseBit2.jpg", phaseBit2)
    save_img(path_pre + "phaseBit3.jpg", phaseBit3)
    save_img(path_pre + "phaseBit4.jpg", phaseBit4)
    save_img(path_pre + "pattern1.jpg", pattern1)
    save_img(path_pre + "pattern2.jpg", pattern2)
    save_img(path_pre + "pattern3.jpg", pattern3)
    save_img(path_pre + "pattern4.jpg", pattern4)
    save_img(path_pre + "phase_mix.jpg", phaseBit_mix)       # 几何分区法 -- 结果码阵
    save_img(path_pre + "pattern_mix.jpg", patternBit_mix)   # 几何分区法 -- 结果码阵方向图
    save_img_xyz(path_pre + "patternBit_mix_xyz.jpg", np.abs(patternBit_mix_xyz), x, y)
    # 保存相位结果
    save_csv(phase1, path_pre + "phase1.csv")
    save_csv(phase2, path_pre + "phase2.csv")
    save_csv(phase3, path_pre + "phase3.csv")
    save_csv(phase4, path_pre + "phase4.csv")
    save_csv(phaseBit1, path_pre + "phaseBit1.csv")
    save_csv(phaseBit2, path_pre + "phaseBit2.csv")
    save_csv(phaseBit3, path_pre + "phaseBit3.csv")
    save_csv(phaseBit4, path_pre + "phaseBit4.csv")
    save_csv(phaseBit_mix, path_pre + "phase_mix.csv")
    # 保存分区结果
    rows, cols = phaseBit1.shape
    phase_cub_1 = np.full((rows, cols), 0)
    phase_cub_2 = np.full((rows, cols), 90)
    phase_cub_3 = np.full((rows, cols), 180)
    phase_cub_4 = np.full((rows, cols), 270)
    phase_cub = nn_ps_beam_4(phase_cub_1, phase_cub_2, phase_cub_3, phase_cub_4, bit_num, buffer_percent)
    save_img(path_pre + "phase_cub.jpg", phase_cub)


def main_multi_beam_8_cub(theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4,
                          theta5, phi5, theta6, phi6, theta7, phi7, theta8, phi8,
                          path_pre, bit_num, buffer_percent):
    logger.info("main_multi_beam_8_cub: bit_num=%d, path_pre=%s, " % (bit_num, path_pre))
    logger.info("main_multi_beam_8_cub: theta1=%d, phi1=%d, theta2=%d, phi2=%d, theta3=%d, phi3=%d, theta4=%d, phi4=%d, "
                "theta5=%d, phi5=%d, theta6=%d, phi6=%d, theta7=%d, phi7=%d, theta8=%d, phi8=%d"
                % (theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4,
                   theta5, phi5, theta6, phi6, theta7, phi7, theta8, phi8))
    # 目前只支持2bit
    if bit_num > 2:
        logger.error("main_multi_beam_8_cub: bit_num bigger than 2.")
        return
    # 获取所有的 phaseBit 变量
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1, bit_num)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2, bit_num)
    phase3, phaseBit3, pattern3 = point_2_phi_pattern(theta3, phi3, bit_num)
    phase4, phaseBit4, pattern4 = point_2_phi_pattern(theta4, phi4, bit_num)
    phase5, phaseBit5, pattern5 = point_2_phi_pattern(theta5, phi5, bit_num)
    phase6, phaseBit6, pattern6 = point_2_phi_pattern(theta6, phi6, bit_num)
    phase7, phaseBit7, pattern7 = point_2_phi_pattern(theta7, phi7, bit_num)
    phase8, phaseBit8, pattern8 = point_2_phi_pattern(theta8, phi8, bit_num)
    # 确保所有数组具有相同的形状
    assert phaseBit1.shape == phaseBit2.shape == phaseBit3.shape == phaseBit4.shape == \
           phaseBit5.shape == phaseBit6.shape == phaseBit7.shape == phaseBit8.shape, "所有数组必须具有相同的形状"
    # NN - PS
    phase_mix = nn_ps_beam_8_cub(phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, bit_num, buffer_percent)
    # 相位转换 X bit
    phaseBit_mix, phaseBitDeg_mix = phase_2_bit(phase_mix, bit_num)
    # 计算phase_mix的方向图
    phaseBit_mix = np.deg2rad(phaseBitDeg_mix)
    # 计算phase_mix
    patternBit_mix = phase_2_pattern(phaseBit_mix)
    # 保存结果
    logger.info("save NN multi-beam 8 result...")
    patternBit_mix_xyz, x, y, z = phase_2_pattern_xyz(phaseBit_mix)
    # 保存图片
    save_img(path_pre + "phase1.jpg", phase1)
    save_img(path_pre + "phase2.jpg", phase2)
    save_img(path_pre + "phase3.jpg", phase3)
    save_img(path_pre + "phase4.jpg", phase4)
    save_img(path_pre + "phase5.jpg", phase5)
    save_img(path_pre + "phase6.jpg", phase6)
    save_img(path_pre + "phase7.jpg", phase7)
    save_img(path_pre + "phase8.jpg", phase8)
    save_img(path_pre + "phaseBit1.jpg", phaseBit1)
    save_img(path_pre + "phaseBit2.jpg", phaseBit2)
    save_img(path_pre + "phaseBit3.jpg", phaseBit3)
    save_img(path_pre + "phaseBit4.jpg", phaseBit4)
    save_img(path_pre + "phaseBit5.jpg", phaseBit5)
    save_img(path_pre + "phaseBit6.jpg", phaseBit6)
    save_img(path_pre + "phaseBit7.jpg", phaseBit7)
    save_img(path_pre + "phaseBit8.jpg", phaseBit8)
    save_img(path_pre + "pattern1.jpg", pattern1)
    save_img(path_pre + "pattern2.jpg", pattern2)
    save_img(path_pre + "pattern3.jpg", pattern3)
    save_img(path_pre + "pattern4.jpg", pattern4)
    save_img(path_pre + "pattern5.jpg", pattern5)
    save_img(path_pre + "pattern6.jpg", pattern6)
    save_img(path_pre + "pattern7.jpg", pattern7)
    save_img(path_pre + "pattern8.jpg", pattern8)
    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_mix)  # 几何分区法 -- 结果码阵
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)  # 几何分区法 -- 结果码阵方向图
    save_img_xyz(path_pre + "patternBit_mix_xyz.jpg", np.abs(patternBit_mix_xyz), x, y)
    # 保存相位结果
    save_csv(phase1, path_pre + "phase1.csv")
    save_csv(phase2, path_pre + "phase2.csv")
    save_csv(phase3, path_pre + "phase3.csv")
    save_csv(phase4, path_pre + "phase4.csv")
    save_csv(phase5, path_pre + "phase5.csv")
    save_csv(phase6, path_pre + "phase6.csv")
    save_csv(phase7, path_pre + "phase7.csv")
    save_csv(phase8, path_pre + "phase8.csv")
    save_csv(phaseBit1, path_pre + "phaseBit1.csv")
    save_csv(phaseBit2, path_pre + "phaseBit2.csv")
    save_csv(phaseBit3, path_pre + "phaseBit3.csv")
    save_csv(phaseBit4, path_pre + "phaseBit4.csv")
    save_csv(phaseBit5, path_pre + "phaseBit5.csv")
    save_csv(phaseBit6, path_pre + "phaseBit6.csv")
    save_csv(phaseBit7, path_pre + "phaseBit7.csv")
    save_csv(phaseBit8, path_pre + "phaseBit8.csv")
    save_csv(phaseBit_mix, path_pre + "phaseBit_mix.csv")
    # 保存分区结果
    rows, cols = phaseBit1.shape
    phase_cub_1 = np.full((rows, cols), 0)
    phase_cub_2 = np.full((rows, cols), 45)
    phase_cub_3 = np.full((rows, cols), 90)
    phase_cub_4 = np.full((rows, cols), 135)
    phase_cub_5 = np.full((rows, cols), 180)
    phase_cub_6 = np.full((rows, cols), 225)
    phase_cub_7 = np.full((rows, cols), 270)
    phase_cub_8 = np.full((rows, cols), 315)
    phase_cub = nn_ps_beam_8_cub(phase_cub_1, phase_cub_2, phase_cub_3, phase_cub_4,
                                 phase_cub_5, phase_cub_6, phase_cub_7, phase_cub_8, bit_num, buffer_percent)
    save_img(path_pre + "phase_cub.jpg", phase_cub)


def main_multi_beam_16_cub(theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4,
                           theta5, phi5, theta6, phi6, theta7, phi7, theta8, phi8,
                           theta9, phi9, theta10, phi10, theta11, phi11, theta12, phi12,
                           theta13, phi13, theta14, phi14, theta15, phi15, theta16, phi16,
                           path_pre, bit_num):
    logger.info("main_multi_beam_16_cub: bit_num=%d, path_pre=%s, " % (bit_num, path_pre))
    logger.info(
        "main_multi_beam_16_cub: theta1=%d, phi1=%d, theta2=%d, phi2=%d, theta3=%d, phi3=%d, theta4=%d, phi4=%d, "
        "theta5=%d, phi5=%d, theta6=%d, phi6=%d, theta7=%d, phi7=%d, theta8=%d, phi8=%d"
        % (theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4,
           theta5, phi5, theta6, phi6, theta7, phi7, theta8, phi8))
    logger.info(
        "main_multi_beam_16_cub: theta9=%d, phi9=%d, theta10=%d, phi10=%d, theta11=%d, phi11=%d, theta12=%d, phi12=%d, "
        "theta13=%d, phi13=%d, theta14=%d, phi14=%d, theta15=%d, phi15=%d, theta16=%d, phi16=%d"
        % (theta9, phi9, theta10, phi10, theta11, phi11, theta12, phi12, theta13, phi13, theta14, phi14,
           theta15, phi15, theta16, phi16))

    # 目前只支持2bit
    if bit_num > 2:
        logger.error("main_multi_beam_16_cub: bit_num bigger than 2.")
        return

    # 获取所有的 phaseBit 变量
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1, bit_num)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2, bit_num)
    phase3, phaseBit3, pattern3 = point_2_phi_pattern(theta3, phi3, bit_num)
    phase4, phaseBit4, pattern4 = point_2_phi_pattern(theta4, phi4, bit_num)
    phase5, phaseBit5, pattern5 = point_2_phi_pattern(theta5, phi5, bit_num)
    phase6, phaseBit6, pattern6 = point_2_phi_pattern(theta6, phi6, bit_num)
    phase7, phaseBit7, pattern7 = point_2_phi_pattern(theta7, phi7, bit_num)
    phase8, phaseBit8, pattern8 = point_2_phi_pattern(theta8, phi8, bit_num)
    phase9, phaseBit9, pattern9 = point_2_phi_pattern(theta9, phi9, bit_num)
    phase10, phaseBit10, pattern10 = point_2_phi_pattern(theta10, phi10, bit_num)
    phase11, phaseBit11, pattern11 = point_2_phi_pattern(theta11, phi11, bit_num)
    phase12, phaseBit12, pattern12 = point_2_phi_pattern(theta12, phi12, bit_num)
    phase13, phaseBit13, pattern13 = point_2_phi_pattern(theta13, phi13, bit_num)
    phase14, phaseBit14, pattern14 = point_2_phi_pattern(theta14, phi14, bit_num)
    phase15, phaseBit15, pattern15 = point_2_phi_pattern(theta15, phi15, bit_num)
    phase16, phaseBit16, pattern16 = point_2_phi_pattern(theta16, phi16, bit_num)

    # 确保所有数组具有相同的形状
    assert phaseBit1.shape == phaseBit2.shape == phaseBit3.shape == phaseBit4.shape == \
           phaseBit5.shape == phaseBit6.shape == phaseBit7.shape == phaseBit8.shape == \
           phaseBit9.shape == phaseBit10.shape == phaseBit11.shape == phaseBit12.shape == \
           phaseBit13.shape == phaseBit14.shape == phaseBit15.shape == phaseBit16.shape, "所有数组必须具有相同的形状"

    # 获取数组的形状
    rows, cols = phaseBit1.shape

    # 创建 phase_mix 数组
    phaseBit_mix = np.zeros((rows, cols))

    # 填充 phase_mix 的16个部分
    phaseBit_mix[:rows // 4, :cols // 4] = phaseBit1[:rows // 4, :cols // 4]
    phaseBit_mix[:rows // 4, cols // 4:cols // 2] = phaseBit2[:rows // 4, cols // 4:cols // 2]
    phaseBit_mix[:rows // 4, cols // 2:3 * cols // 4] = phaseBit3[:rows // 4, cols // 2:3 * cols // 4]
    phaseBit_mix[:rows // 4, 3 * cols // 4:] = phaseBit4[:rows // 4, 3 * cols // 4:]

    phaseBit_mix[rows // 4:rows // 2, :cols // 4] = phaseBit5[rows // 4:rows // 2, :cols // 4]
    phaseBit_mix[rows // 4:rows // 2, cols // 4:cols // 2] = phaseBit6[rows // 4:rows // 2, cols // 4:cols // 2]
    phaseBit_mix[rows // 4:rows // 2, cols // 2:3 * cols // 4] = phaseBit7[rows // 4:rows // 2, cols // 2:3 * cols // 4]
    phaseBit_mix[rows // 4:rows // 2, 3 * cols // 4:] = phaseBit8[rows // 4:rows // 2, 3 * cols // 4:]

    phaseBit_mix[rows // 2:3 * rows // 4, :cols // 4] = phaseBit9[rows // 2:3 * rows // 4, :cols // 4]
    phaseBit_mix[rows // 2:3 * rows // 4, cols // 4:cols // 2] = phaseBit10[rows // 2:3 * rows // 4,
                                                                 cols // 4:cols // 2]
    phaseBit_mix[rows // 2:3 * rows // 4, cols // 2:3 * cols // 4] = phaseBit11[rows // 2:3 * rows // 4,
                                                                     cols // 2:3 * cols // 4]
    phaseBit_mix[rows // 2:3 * rows // 4, 3 * cols // 4:] = phaseBit12[rows // 2:3 * rows // 4, 3 * cols // 4:]

    phaseBit_mix[3 * rows // 4:, :cols // 4] = phaseBit13[3 * rows // 4:, :cols // 4]
    phaseBit_mix[3 * rows // 4:, cols // 4:cols // 2] = phaseBit14[3 * rows // 4:, cols // 4:cols // 2]
    phaseBit_mix[3 * rows // 4:, cols // 2:3 * cols // 4] = phaseBit15[3 * rows // 4:, cols // 2:3 * cols // 4]
    phaseBit_mix[3 * rows // 4:, 3 * cols // 4:] = phaseBit16[3 * rows // 4:, 3 * cols // 4:]

    # 计算 phase_mix 的方向图
    if bit_num == 1:
        # 1bit相位
        phaseBit_mix = np.deg2rad(phaseBit_mix * 180)
    elif bit_num == 2:
        # 2bit相位
        phaseBit_mix = np.deg2rad(phaseBit_mix * 45)
    else:
        # 默认1bit相位
        phaseBit_mix = np.deg2rad(phaseBit_mix * 180)

    patternBit_mix = phase_2_pattern(phaseBit_mix)

    # 保存结果
    logger.info("save NN multi-beam 16 result...")
    patternBit_mix_xyz, x, y, z = phase_2_pattern_xyz(phaseBit_mix)

    # 保存图片
    for i in range(1, 17):
        save_img(path_pre + f"phase{i}.jpg", eval(f"phase{i}"))
        save_img(path_pre + f"phaseBit{i}.jpg", eval(f"phaseBit{i}"))
        save_img(path_pre + f"pattern{i}.jpg", eval(f"pattern{i}"))

    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_mix)  # 几何分区法 -- 结果码阵
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)  # 几何分区法 -- 结果码阵方向图
    save_img_xyz(path_pre + "patternBit_mix_xyz.jpg", np.abs(patternBit_mix_xyz), x, y)

    # 保存相位结果
    for i in range(1, 17):
        save_csv(eval(f"phase{i}"), path_pre + f"phase{i}.csv")
        save_csv(eval(f"phaseBit{i}"), path_pre + f"phaseBit{i}.csv")

    save_csv(phaseBit_mix, path_pre + "phaseBit_mix.csv")




if __name__ == '__main__':
    # 配置日志，默认打印到控制台，也可以设置打印到文件
    setup_logging()
    # setup_logging(log_file="../../files/logs/log_multi_beam_NN.log")

    # 获取日志记录器并记录日志
    logger = logging.getLogger("[RIS-multi-beam-NN-PS]")
    logger.info("1bit-RIS-multi-beam-NN-PS: NN + PS")
    # 几何分区法: 主函数
    main_multi_beam_2(30, 0, 30, 90,
                      "../files/multi-beam/1bit/NN-PS/2-(30,0,30,90)/", 1, 3)
    main_multi_beam_2(30, 0, 30, 180,
                      "../files/multi-beam/1bit/NN-PS/2-(30,0,30,180)/", 1, 3)
    main_multi_beam_4(30, 0, 30, 60, 30, 120, 30, 180,
                      "../files/multi-beam/1bit/NN-PS/4-(30,0,30,60,30,120,30,180)/", 1, 3)
    main_multi_beam_4(30, 0, 30, 90, 30, 180, 30, 270,
                      "../files/multi-beam/1bit/NN-PS/4-(30,0,30,90,30,180,30,270)/", 1, 3)
    main_multi_beam_8_cub(30, 0, 30, 45, 30, 90, 30, 135, 30, 180, 30, 225, 30, 270, 30, 315,
                          "../files/multi-beam/1bit/NN-PS/", 1, 3)
    # main_multi_beam_16_cub(30, 0, 30, 22.5, 30, 45, 30, 67.5, 30, 90, 30, 112.5, 30, 135, 30, 157.5,
    #                        30, 180, 30, 202.5, 30, 225, 30, 247.5, 30, 270, 30, 292.5, 30, 315, 30, 337.5,
    #                        "../../files/multi-beam/2bit/NN/16-cub-(30,22.5step)/", 2)
    # test
    # test_split_array()
    # test_phaseBit_mix_16()
    # test_phaseBit_mix_8()
    # test_phaseBit_mix_2()