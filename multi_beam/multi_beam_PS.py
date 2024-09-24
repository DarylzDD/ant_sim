import logging
import numpy as np

from util.util_log import setup_logging
from util.util_csv import save_csv
from util.util_image import save_img, save_img_xyz
from util.util_ris_pattern import point_2_phi_pattern, phase_2_pattern, phase_2_pattern_xyz, eps, phase_2_bit
from util.util_statistics import calculate_statistics
from util.util_analysis_plane import get_peaks, get_peak_3rd


# 生成指向
def generate_points(theta0, phi0, phi_step, phi_max):
    points = []
    current_phi = phi0
    #
    while current_phi < phi_max:
        points.append([theta0, current_phi])
        current_phi += phi_step
    #
    return points


# ============================================= ris的"1"随机取150°~200° ====================================
# 统计N次1bit中"1"随机取角度的PSLL, 阵列, 方向图
def multi_beam_2_rand(iteration, phaseBit_mix):
    list_psll = []
    best_psll = 0
    best_phaseBit = None
    best_phaseBitDeg = None
    best_patternBit = None
    worst_psll = -200
    worst_phaseBit = None
    worst_phaseBitDeg = None
    worst_patternBit = None
    #
    for i in range(iteration):
        # phaseBit_mix_rand: 连续相位转1bit相位, 1时随机取150°~200°
        random_degrees = np.random.randint(150, 201, size=phaseBit_mix.shape)
        phaseBit_mix_rand_deg = phaseBit_mix * random_degrees
        phaseBit_mix_rand = np.deg2rad(phaseBit_mix_rand_deg)
        # 计算方向图 pattern 和 pattern_dbw
        patternBit_mix_rand = phase_2_pattern(phaseBit_mix_rand)
        patternBit_mix_rand_dbw = 20 * np.log10(
            np.abs(patternBit_mix_rand) / np.max(np.max(np.abs(patternBit_mix_rand))) + eps)
        # 计算PSLL
        peaks = get_peaks(patternBit_mix_rand_dbw)
        peaks_3rd = get_peak_3rd(peaks)
        # 保存每次的PSLL, 最好和最差的阵列和方向图
        if peaks_3rd is not None:
            peaks_3rd_val = peaks_3rd[0]
            list_psll.append(peaks_3rd_val)
            if peaks_3rd_val > worst_psll:
                worst_psll = peaks_3rd_val
                worst_phaseBit = phaseBit_mix_rand
                worst_phaseBitDeg = phaseBit_mix_rand_deg
                worst_patternBit = patternBit_mix_rand
            if peaks_3rd_val < best_psll:
                best_psll = peaks_3rd_val
                best_phaseBit = phaseBit_mix_rand
                best_phaseBitDeg = phaseBit_mix_rand_deg
                best_patternBit = patternBit_mix_rand
        logger.info("i=%d, peaks_3rd=%s, best_psll=%f, worst_psll=%f" % (i, peaks_3rd, best_psll, worst_psll))
    return list_psll, best_psll, best_phaseBit, best_phaseBitDeg, best_patternBit, \
           worst_psll, worst_phaseBit, worst_phaseBitDeg, worst_patternBit


# ============================================= 主函数 ====================================
# 相位合成法 -- 双波束 -- 核心方法
def psm_beam_2(phase1, phase2, bit_num=1):
    # 确保 phase1 和 phase2 具有相同的形状
    assert phase1.shape == phase1.shape, "phase1 和 phase2 必须具有相同的形状"
    # 计算 phase1 和 phase2 的差的绝对值
    diff = np.abs(phase1 - phase2)
    # 根据条件创建 phase_mix
    phase_mix = np.where(diff <= 180, (phase1 + phase2) / 2,
                         np.where((diff > 180) & (diff <= 360), (phase1 + phase2 + 360) / 2, 0))
    # 相位转换 X bit
    phaseBit_mix, phaseBitDeg_mix = phase_2_bit(phase_mix, bit_num)
    #
    return phase_mix, phaseBit_mix, phaseBitDeg_mix


# 相位合成法 -- 四波束 -- 核心方法
def psm_beam_4(phase1, phase2, phase3, phase4, bit_num=1):
    # 第一步：计算 phase1 和 phase2 的差值 diff1
    diff1 = np.abs(phase1 - phase2)
    # 根据 diff1 更新 phase_mix
    phase_mix = np.where(diff1 <= 180, (phase1 + phase2) / 2, (phase1 + phase2 + 360) / 2)
    # 第二步：计算 phase_mix 和 phase3 的差值 diff2
    diff2 = np.abs(phase_mix - phase3)
    # 根据 diff2 更新 phase_mix
    phase_mix = np.where(diff2 <= 180, (phase_mix + phase3) / 2, (phase_mix + phase3 + 360) / 2)
    # 第三步：计算 phase_mix 和 phase4 的差值 diff3
    diff3 = np.abs(phase_mix - phase4)
    # 根据 diff3 更新 phase_mix
    phase_mix = np.where(diff3 <= 180, (phase_mix + phase4) / 2, (phase_mix + phase4 + 360) / 2)
    # 相位转换 X bit
    phaseBit_mix, phaseBitDeg_mix = phase_2_bit(phase_mix, bit_num)
    #
    return phase_mix, phaseBit_mix, phaseBitDeg_mix


# 相位合成法 -- N波束 -- 核心方法
def psm_beam_n(phases, bit_num=1):
    # 初始化 phase_mix 为第一个波束相位的副本
    phase_mix = np.copy(phases[0])
    # 遍历每个后续的相位
    for phase in phases[1:]:
        # 计算 phase_mix 和当前 phase 的差值
        diff = np.abs(phase_mix - phase)
        # 根据差值更新 phase_mix
        phase_mix = np.where(diff <= 180, (phase_mix + phase) / 2, (phase_mix + phase + 360) / 2)
    # 相位转换 X bit
    phaseBit_mix, phaseBitDeg_mix = phase_2_bit(phase_mix, bit_num)
    return phase_mix, phaseBit_mix, phaseBitDeg_mix


# 相位合成法 -- 双波束
def main_multi_beam_2(theta1, phi1, theta2, phi2, path_pre, bit_num):
    logger.info("main_multi_beam_2: bit_num=%d, path_pre=%s, " % (bit_num, path_pre))
    logger.info("main_multi_beam_2: theta1=%d, phi1=%d, theta2=%d, phi2=%d, " % (theta1, phi1, theta2, phi2))
    # 目前只支持2bit
    if bit_num > 2:
        logger.error("main_multi_beam_2: bit_num bigger than 2.")
        return
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1, bit_num)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2, bit_num)
    # 相位合成法 核心方法
    phase_mix, phaseBit_mix, phaseBitDeg_mix = psm_beam_2(phase1, phase2, bit_num)
    # 计算phase_mix的方向图
    phase_mix = np.deg2rad(phase_mix)
    phaseBit_mix = np.deg2rad(phaseBitDeg_mix)
    pattern_mix = phase_2_pattern(phase_mix)
    patternBit_mix = phase_2_pattern(phaseBit_mix)
    patternBit_mix_xyz, x, y, z  = phase_2_pattern_xyz(phaseBit_mix)
    #
    # 保存结果
    logger.info("save PS multi-beam 2 result...")
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
    save_img_xyz(path_pre + "patternBit_mix_xyz.jpg", np.abs(patternBit_mix_xyz), x, y)
    # 保存相位结果
    save_csv(phase1, path_pre + "phase1.csv")
    save_csv(phase2, path_pre + "phase2.csv")
    save_csv(phaseBit1, path_pre + "phaseBit1.csv")
    save_csv(phaseBit2, path_pre + "phaseBit2.csv")
    save_csv(phase_mix, path_pre + "phase_mix.csv")
    save_csv(phaseBit_mix, path_pre + "phaseBit_mix.csv")


# 相位叠加法 -- 双波束 (但是1bit相位非0或1，随机生成)
def main_multi_beam_2_random(theta1, phi1, theta2, phi2, path_pre):
    logger.info("main_multi_beam_2_random: theta1=%d, phi1=%d, theta2=%d, phi2=%d, " % (theta1, phi1, theta2, phi2))
    # 根据波束指向, 计算阵列(连续角度)
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2)
    # 相位合成法 核心方法
    phase_mix, phaseBit_mix, phaseBitDeg_mix = psm_beam_2(phase1, phase2)
    # 统计30次1bit中"1"随机取角度的PSLL, 阵列, 方向图
    list_psll, best_psll, best_phaseBit, best_phaseBitDeg, best_patternBit, \
    worst_psll, worst_phaseBit, worst_phaseBitDeg, worst_patternBit = multi_beam_2_rand(30, phaseBit_mix)
    # 计算统计参数
    psll_max, psll_min, psll_ave, psll_std = calculate_statistics(list_psll)
    logger.info("psll_max=%f, psll_min=%f, psll_ave=%f, psll_std=%f" % (psll_max, psll_min, psll_ave, psll_std))
    # 保存结果
    logger.info("save NN multi-beam 2 (random) result...")
    phase_mix = np.deg2rad(phase_mix)
    phaseBit_mix = np.deg2rad(phaseBit_mix * 180)
    pattern_mix = phase_2_pattern(phase_mix)
    patternBit_mix = phase_2_pattern(phaseBit_mix)
    patternBit_mix_xyz, x, y, z = phase_2_pattern_xyz(phaseBit_mix)
    patternBit_mix_xyz_best, x_best, y_best, z_best = phase_2_pattern_xyz(best_phaseBit)
    patternBit_mix_xyz_worst, x_worst, y_worst, z_worst = phase_2_pattern_xyz(worst_phaseBit)
    # 计算不取随机"1"的PSLL
    patternBit_mix_dbw = 20 * np.log10(
        np.abs(patternBit_mix) / np.max(np.max(np.abs(patternBit_mix))) + eps)
    peaks = get_peaks(patternBit_mix_dbw)
    peaks_3rd = get_peak_3rd(peaks)
    psll_mix_bit = peaks_3rd[0]
    # 保存图片
    save_img(path_pre + "phase1.jpg", phase1)
    save_img(path_pre + "phase2.jpg", phase2)
    save_img(path_pre + "phaseBit1.jpg", phaseBit1)
    save_img(path_pre + "phaseBit2.jpg", phaseBit2)
    save_img(path_pre + "pattern1.jpg", pattern1)
    save_img(path_pre + "pattern2.jpg", pattern2)
    save_img(path_pre + "phase_mix.jpg", phase_mix)
    save_img(path_pre + "pattern_mix.jpg", pattern_mix)
    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_mix)         # 几何分区法 -- 结果码阵
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)     # 几何分区法 -- 结果码阵方向图
    save_img(path_pre + "best_patternBit.jpg", best_patternBit)
    save_img(path_pre + "worst_patternBit.jpg", worst_patternBit)
    save_img(path_pre + "best_phaseBit.jpg", best_phaseBit)
    save_img(path_pre + "worst_phaseBit.jpg", worst_phaseBit)
    save_img_xyz(path_pre + "patternBit_mix_xyz.jpg", np.abs(patternBit_mix_xyz), x, y)
    save_img_xyz(path_pre + "patternBit_mix_xyz_best.jpg", np.abs(patternBit_mix_xyz_best), x_best, y_best)
    save_img_xyz(path_pre + "patternBit_mix_xyz_worst.jpg", np.abs(patternBit_mix_xyz_worst), x_worst, y_worst)
    # 保存相位结果
    save_csv(phase1, path_pre + "phase1.csv")
    save_csv(phase2, path_pre + "phase2.csv")
    save_csv(phaseBit1, path_pre + "phaseBit1.csv")
    save_csv(phaseBit2, path_pre + "phaseBit2.csv")
    save_csv(phase_mix, path_pre + "phase_mix.csv")
    save_csv(phaseBit_mix, path_pre + "phaseBit_mix.csv")
    save_csv(best_phaseBit, path_pre + "best_phaseBit.csv")
    save_csv(worst_phaseBit, path_pre + "worst_phaseBit.csv")
    save_csv(best_phaseBitDeg, path_pre + "best_phaseBitDeg.csv")
    save_csv(worst_phaseBitDeg, path_pre + "worst_phaseBitDeg.csv")
    # 保存统计结果
    with open(path_pre + 'psll_res.txt', 'a') as file:
        file.write("(theta1, phi1)=(%f, %f), (theta2, phi2)=(%f, %f)\n" % (theta1, phi1, theta2, phi2))
        file.write("psll_mix_bit=%f\n" % (psll_mix_bit))
        file.write("best_psll=%f, worst_psll=%f\n" % (best_psll, worst_psll))
        file.write("psll_max=%f, psll_min=%f, psll_ave=%f, psll_std=%f\n" % (psll_max, psll_min, psll_ave, psll_std))
        file.write("list_psll=%s\n" % list_psll)


# 相位合成法 -- 四波束
def main_multi_beam_4(theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4, path_pre, bit_num):
    logger.info("main_multi_beam_4: bit_num=%d, path_pre=%s, " % (bit_num, path_pre))
    logger.info("main_multi_beam_4: theta1=%d, phi1=%d, theta2=%d, phi2=%d, theta3=%d, phi3=%d, theta4=%d, phi4=%d"
                % (theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4))
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1, bit_num)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2, bit_num)
    phase3, phaseBit3, pattern3 = point_2_phi_pattern(theta3, phi3, bit_num)
    phase4, phaseBit4, pattern4 = point_2_phi_pattern(theta4, phi4, bit_num)
    # 确保所有数组具有相同的形状
    assert phaseBit1.shape == phaseBit2.shape == phaseBit3.shape == phaseBit4.shape, "所有数组必须具有相同的形状"

    # case-1
    # # 计算所有差值的绝对值
    # diff12 = np.abs(phase1 - phase2)
    # diff13 = np.abs(phase1 - phase3)
    # diff14 = np.abs(phase1 - phase4)
    # diff23 = np.abs(phase2 - phase3)
    # diff24 = np.abs(phase2 - phase4)
    # diff34 = np.abs(phase3 - phase4)
    # # 检查所有差值的绝对值是否都小于等于180
    # all_diff_lte_180 = (diff12 <= 180) & (diff13 <= 180) & (diff14 <= 180) & (diff23 <= 180) & (diff24 <= 180) & (
    #             diff34 <= 180)
    # # 检查是否有差值的绝对值大于180且小于等于360
    # any_diff_gt_180_lte_360 = ((diff12 > 180) & (diff12 <= 360)) | ((diff13 > 180) & (diff13 <= 360)) | (
    #             (diff14 > 180) & (diff14 <= 360)) | \
    #                           ((diff23 > 180) & (diff23 <= 360)) | ((diff24 > 180) & (diff24 <= 360)) | (
    #                                       (diff34 > 180) & (diff34 <= 360))
    # 计算 phase_mix
    # phase_mix = np.where(all_diff_lte_180, (phase1 + phase2 + phase3 + phase4) / 4,
    #                      np.where(any_diff_gt_180_lte_360, (phase1 + phase2 + phase3 + phase4 + 360) / 4, 0))

    # case-2
    # # 计算所有差值的绝对值
    # diff12 = np.abs(phase1 - phase2)
    # diff13 = np.abs(phase1 - phase3)
    # diff14 = np.abs(phase1 - phase4)
    # diff23 = np.abs(phase2 - phase3)
    # diff24 = np.abs(phase2 - phase4)
    # diff34 = np.abs(phase3 - phase4)
    # # 检查是否有差值的绝对值小于等于180
    # any_diff_lte_180 = (diff12 <= 180) | (diff13 <= 180) | (diff14 <= 180) | (diff23 <= 180) | (diff24 <= 180) | (
    #             diff34 <= 180)
    # # 检查所有差值的绝对值是否都大于180且小于等于360
    # all_diff_gt_180_lte_360 = (diff12 > 180) & (diff12 <= 360) & (diff13 > 180) & (diff13 <= 360) & (diff14 > 180) & (
    #             diff14 <= 360) & \
    #                           (diff23 > 180) & (diff23 <= 360) & (diff24 > 180) & (diff24 <= 360) & (diff34 > 180) & (
    #                                       diff34 <= 360)
    # phase_mix = np.where(any_diff_lte_180, (phase1 + phase2 + phase3 + phase4) / 4,
    #                      np.where(all_diff_gt_180_lte_360, (phase1 + phase2 + phase3 + phase4 + 360) / 4, 0))

    # case-3
    phase_mix, phaseBit_mix, phaseBitDeg_mix = psm_beam_4(phase1, phase2, phase3, phase4, bit_num)

    # 计算phase_mix的方向图
    phase_mix = np.deg2rad(phase_mix)
    phaseBit_mix = np.deg2rad(phaseBitDeg_mix)
    pattern_mix = phase_2_pattern(phase_mix)
    patternBit_mix = phase_2_pattern(phaseBit_mix)
    #
    # 保存结果
    logger.info("save PS multi-beam 4 result...")
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
    save_img(path_pre + "phase_mix.jpg", phase_mix)
    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_mix)  # 相位合成法 -- 结果码阵
    save_img(path_pre + "pattern_mix.jpg", pattern_mix)
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)  # 相位合成法 -- 结果码阵方向图
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
    save_csv(phase_mix, path_pre + "phase_mix.csv")
    save_csv(phaseBit_mix, path_pre + "phaseBit_mix.csv")


# 相位合成法 -- N波束
def main_multi_beam_N(points, path_pre, bit_num):
    logger.info("main_multi_beam_N: bit_num=%d, path_pre=%s, " % (bit_num, path_pre))
    logger.info("main_multi_beam_N: num of points = %d" % (len(points)))
    logger.info("main_multi_beam_N: points = %s" % (points))
    phase_pattern_list = []
    phase_list = []
    for point in points:
        theta = point[0]
        phi = point[1]
        phase, phaseBit, pattern = point_2_phi_pattern(theta, phi, bit_num)
        phase_pattern_list.append([phase, phaseBit, pattern])
        phase_list.append(phase)
    # 相位合成法
    phase_mix, phaseBit_mix, phaseBitDeg_mix = psm_beam_n(phase_list, bit_num)
    # 计算phase_mix的方向图
    phase_mix = np.deg2rad(phase_mix)
    phaseBit_mix = np.deg2rad(phaseBitDeg_mix)
    pattern_mix = phase_2_pattern(phase_mix)
    patternBit_mix = phase_2_pattern(phaseBit_mix)
    #
    # 保存结果
    logger.info("save NN multi-beam N result...")
    patternBit_mix_xyz, x, y, z = phase_2_pattern_xyz(phaseBit_mix)
    # 保存结果
    for i in range(len(phase_pattern_list)):
        phase = phase_pattern_list[i][0]
        phaseBit = phase_pattern_list[i][1]
        pattern = phase_pattern_list[i][2]
        # 保存图片
        save_img(path_pre + "phase" + str(i+1) + ".jpg", phase)
        save_img(path_pre + "phaseBit" + str(i+1) + ".jpg", phaseBit)
        save_img(path_pre + "pattern" + str(i+1) + ".jpg", pattern)
        # 保存相位结果
        save_csv(phase, path_pre + "phase" + str(i+1) + ".csv")
        save_csv(phaseBit, path_pre + "phaseBit" + str(i+1) + ".csv")
    # 保存图片
    save_img(path_pre + "phase_mix.jpg", phase_mix)       # 相位合成法 -- 结果码阵
    save_img(path_pre + "pattern_mix.jpg", pattern_mix)   # 相位合成法 -- 结果码阵方向图
    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_mix)  # 相位合成法 -- 结果码阵
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)  # 相位合成法 -- 结果码阵方向图
    save_img_xyz(path_pre + "patternBit_mix_xyz.jpg", np.abs(patternBit_mix_xyz), x, y)
    # 保存相位结果
    save_csv(phase_mix, path_pre + "phase_mix.csv")
    save_csv(phaseBit_mix, path_pre + "phaseBit_mix.csv")



if __name__ == '__main__':
    # 配置日志，默认打印到控制台，也可以设置打印到文件
    setup_logging()
    # setup_logging(log_file="../../files/logs/log_multi_beam_PS.log")

    # 获取日志记录器并记录日志
    logger = logging.getLogger("[RIS-multi-beam-PS-1bit]")
    logger.info("1bit-RIS-multi-beam-PS: Geometric partitioning method")
    # 相位合成法: 主函数
    # main_multi_beam_2_random(30, 0, 30, 90, "../../files/multi-beam/PS/rand-2-(30,0,30,90)/")
    # main_multi_beam_2(30, 0, 30, 90, "../../files/multi-beam/2bit/PS/2-(30,0,30,90)/", 2)
    # main_multi_beam_2(30, 0, 30, 180, "../../files/multi-beam/2bit/PS/2-(30,0,30,180)/", 2)
    # main_multi_beam_4(30, 0, 30, 60, 30, 120, 30, 180, "../../files/multi-beam/2bit/PS/4-(30,0,30,60,30,120,30,180)/", 2)
    # main_multi_beam_4(30, 0, 30, 90, 30, 180, 30, 270, "../../files/multi-beam/2bit/PS/4-(30,0,30,90,30,180,30,270)/", 2)
    # main_multi_beam_N([[30, 0], [30, 180]], "../../files/multi-beam/2bit/PS/B-2-(30,0,30,180)/", 2)
    # main_multi_beam_N([[30, 0], [30, 90]], "../../files/multi-beam/2bit/PS/B-2-(30,0,30,90)/", 2)
    # main_multi_beam_N([[30, 0], [30, 60], [30, 120], [30, 180]],
    #                   "../../files/multi-beam/2bit/PS/B-4-(30,0,30,60,30,120,30,180)/", 2)
    # main_multi_beam_N([[30, 0], [30, 90], [30, 180], [30, 270]],
    #                   "../../files/multi-beam/2bit/PS/B-4-(30,0,30,90,30,180,30,270)/", 2)
    main_multi_beam_N(generate_points(30, 0, 45, 360), "../../files/multi-beam/1bit/PS/B-8-uni-(30,45step)/", 1)
    # main_multi_beam_N(generate_points(30, 0, 22.5, 360), "../../files/multi-beam/PS/B-16-uni-(30,22.5step)/")
    # main_multi_beam_N(generate_points(30, 0, 11.25, 360), "../../files/multi-beam/PS/B-32-uni-(30,11.25step)/")
    #
    # test
    # points = generate_points(30, 0, 45, 360)
    # logger.info("point: %s" % (points))