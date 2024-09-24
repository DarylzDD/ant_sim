import logging
import numpy as np
import matplotlib.pyplot as plt

from util.util_log import setup_logging
from util.util_csv import save_csv
from util.util_image import save_img, draw_img_xyz, draw_img
from util.util_ris_pattern import point_2_phi_pattern, phase_2_pattern, point_2_phase, phase_2_pattern_xyz
from util.util_analysis_plane import get_peaks, get_peak_3rd
from util.util_mat import read_mat


# 画频次的图
def draw_map(list1):
    # 给定的二维数组
    # list1 = [[0, 1, 2, 3, 2, 0], [1, 3, 1, 1, 2, 1]]
    # 获取数组的行数和列数
    num_rows = len(list1)
    num_cols = len(list1[0])
    # 创建一个图和网格
    fig, ax = plt.subplots()
    # 隐藏坐标轴
    ax.axis('off')
    # 画出网格中的值
    for i in range(num_rows):
        for j in range(num_cols):
            cell_value = list1[i][j]
            # 在对应的格子里写上数值
            ax.text(j + 0.5, num_rows - i - 0.5, str(cell_value), va='center', ha='center')
    # 画出网格线
    for x in range(num_cols + 1):
        ax.axhline(x, lw=2, color='k', zorder=5)
        ax.axvline(x, lw=2, color='k', zorder=5)
    # 设置坐标轴的显示范围
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    # 反转y轴方向，保证第一个元素在左上角
    ax.invert_yaxis()
    # 显示图形
    plt.show()


# ============================================= 稀疏相关 ====================================
# 获取phase的psll -- 双波束
def get_psll_beam_2(phase):
    pattern = phase_2_pattern(phase)
    #
    eps = 0.0001
    max_p = np.max(np.max(np.abs(pattern)))
    pattern_dbw = 20 * np.log10(np.abs(pattern) / max_p + eps)
    #
    third_peak_val = 0
    peaks = get_peaks(pattern_dbw)
    if len(peaks) > 1:
        peak_3rd = get_peak_3rd(peaks)
        third_peak_val = peak_3rd[0]
    return third_peak_val


# 将当前点和周围点设置为0
def set_phase_0_window(i, j, rows, cols, phase, radius):
    phase_new = phase.copy()
    # 处理当前位置及周围的点
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            # 防止索引越界
            if 0 <= i + di < rows and 0 <= j + dj < cols:
                phase_new[i + di][j + dj] = 0.0
            else:
                return None
    return phase_new


# 返回稀疏矩阵对阵列影响程度的矩阵 -- 遍历周围为0法
def generate_weight_by_sparse_each_close(phase, radius=1, step=1):
    peak_3rd_val = get_psll_beam_2(phase)
    logger.info("peak_3rd_val=%f" % peak_3rd_val)
    # 权重矩阵
    weights = np.zeros_like(phase)
    item = 0
    rows, cols = phase.shape
    for i in range(0, rows, step):
        for j in range(0, cols, step):

            phase_new = set_phase_0_window(i, j, rows, cols, phase, radius)
            if phase_new is None:
                continue
            # draw_img(phase_new)

            peak_3rd_val_new = get_psll_beam_2(phase_new)
            logger.info("i=%d, j=%d, peak_3rd_val=%f, peak_3rd_val_new=%f" % (i, j, peak_3rd_val, peak_3rd_val_new))
            weights[i][j] = peak_3rd_val_new

            logger.info("i=%d, j=%d, item=%d" % (i, j, item))
            item += 1

    save_csv(weights, "../../files/multi-beam/SC/weight/w-(30,270)-8-8.csv")
    return weights


# 返回稀疏矩阵对阵列影响程度的矩阵 -- 遍历阵元为0法
def generate_weight_by_sparse_each(phase):
    peak_3rd_val = get_psll_beam_2(phase)
    logger.info("peak_3rd_val=%f" % peak_3rd_val)
    # 权重矩阵
    weights = np.zeros_like(phase)
    item = 0
    for i in range(phase.shape[0]):
        for j in range(phase.shape[1]):
    # for i in range(0, phase.shape[0], 3):
    #     for j in range(0, phase.shape[1], 3):
            phase_new = phase.copy()
            phase_new[i][j] = 0.0
            peak_3rd_val_new = get_psll_beam_2(phase_new)
            logger.info("i=%d, j=%d, peak_3rd_val=%f, peak_3rd_val_new=%f" % (i, j, peak_3rd_val, peak_3rd_val_new))
            weights[i][j] = peak_3rd_val_new
            #
            logger.info("i=%d, j=%d, item=%d" % (i, j, item))
            item = item + 1
    save_csv(weights, "../../files/multi-beam/SC/weight/w-(30,0).csv")
    return weights


# 返回稀疏矩阵对阵列影响程度的矩阵 -- 独立多次稀疏法
def get_sparse_from_mat(path_file_pre, item_start, item_end, item_step):
    list_data = []
    for i in range(item_start, item_end, item_step):
        path_file = path_file_pre + str(i) + ".mat"
        logger.info("path_file: %s" % path_file)
        data = read_mat(path_file)
        list_data.append(data)
    return list_data


# ============================================= 主函数 ====================================
# 稀疏结果做聚类合成阵列 -- 双波束 -- 核心方法
def sc_beam_2(theta1, phi1, theta2, phi2):
    phase_mix, phaseBit_mix = None, None
    #
    return phase_mix, phaseBit_mix


# 稀疏结果做聚类合成阵列 -- 双波束
def main_multi_beam_2(theta1, phi1, theta2, phi2, path_pre):
    logger.info("main_multi_beam_2: theta1=%d, phi1=%d, theta2=%d, phi2=%d, " % (theta1, phi1, theta2, phi2))
    # 核心方法
    phase_mix, phaseBit_mix = sc_beam_2(theta1, phi1, theta2, phi2)
    # 计算phase_mix的方向图
    pattern_mix = phase_2_pattern(phase_mix)
    patternBit_mix = phase_2_pattern(phaseBit_mix)
    #
    # 保存结果
    logger.info("save PS multi-beam 2 result...")
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2)
    # 保存图片
    save_img(path_pre + "phase1.jpg", phase1)
    save_img(path_pre + "phase2.jpg", phase2)
    save_img(path_pre + "phaseBit1.jpg", phaseBit1)
    save_img(path_pre + "phaseBit2.jpg", phaseBit2)
    save_img(path_pre + "pattern1.jpg", pattern1)
    save_img(path_pre + "pattern2.jpg", pattern2)
    save_img(path_pre + "phase_mix.jpg", phase_mix)
    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_mix)         # 相位合成法 -- 结果码阵
    save_img(path_pre + "pattern_mix.jpg", pattern_mix)
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)     # 相位合成法 -- 结果码阵方向图
    # 保存相位结果
    save_csv(phase1, path_pre + "phase1.csv")
    save_csv(phase2, path_pre + "phase2.csv")
    save_csv(phaseBit1, path_pre + "phaseBit1.csv")
    save_csv(phaseBit2, path_pre + "phaseBit2.csv")
    save_csv(phase_mix, path_pre + "phase_mix.csv")
    save_csv(phaseBit_mix, path_pre + "phaseBit_mix.csv")


# 通过去除一个, 计算稀疏生成权重矩阵
def main_genarate_weight_by_sparse_each():
    theta0 = 30
    phi0 = 270
    phase = point_2_phase(theta0, phi0)
    # generate_weight_by_sparse_each(phase)
    generate_weight_by_sparse_each_close(phase, 8, 8)
    #
    # phase, phaseBit, pattern = point_2_phi_pattern(theta0, phi0)
    # pattern_xyz, x, y, z = phase_2_pattern_xyz(phaseBit)
    # draw_img(pattern)
    # draw_img_xyz(np.abs(pattern_xyz), x, y)


# 通过多次独立稀疏结果计算频次, 生成权重矩阵
def main_genarate_weight_by_sparse():
    # 1.读取稀疏结果
    list_datas = get_sparse_from_mat("../../files/chebyshev_thin_plane_64(30,0)/fBest", 1800, 4000, 50)
    # for datas in list_datas:
    #     print(datas)
    # 2.相加每次稀疏结果, 得到阵元在稀疏结果中出现的频次
    result = [sum(x) for x in zip(*list_datas)]
    # print("result:")
    # 打印宽度为3的列表下标
    # print("[{}]".format(", ".join("{:<4}".format(i) for i, _ in enumerate(result))))
    # 打印宽度为3的列表值
    # print("[{}]".format(", ".join("{:<4}".format(i) for i in result)))
    # print(result)
    # 3.排序并得到最大频次
    result_sorted = sorted(result, reverse=True)
    result_max = result_sorted[0]
    # print("result_max:", result_max)
    # print("result_min:", result_sorted[-1])
    # debug: 画个图看看对不对
    x, y = 64, 64  # 设定行数和列数
    result2 = [result[i * y:(i + 1) * y] for i in range(x)]
    # print("result2:")
    # print(result2)
    draw_map(result2)
    draw_img(result2)
    # 4.计算权重
    weight_list = []
    for r in result:
        weight_list.append(float(r / result_max))
    return weight_list




if __name__ == '__main__':
    # 配置日志，默认打印到控制台，也可以设置打印到文件
    setup_logging()
    # setup_logging(log_file="../../files/logs/log_multi_beam_SC.log")
    #
    # 获取日志记录器并记录日志
    logger = logging.getLogger("[RIS-multi-beam-SC-1bit]")
    logger.info("1bit-RIS-multi-beam-SC: Sparse & Cluster")
    # 相位合成法: 主函数
    # main_multi_beam_2(30, 0, 30, 90, "../../files/multi-beam/SC/2-(30,0,30,90)/")
    # 计算权重方式 1: 删除某一阵元后, 计算 psll 做权重矩阵, 结论是删除某一阵元似乎差不多
    main_genarate_weight_by_sparse_each()
    # 计算权重方式 2: 用独立多次稀疏之后结果频次做权重
    # main_genarate_weight_by_sparse()