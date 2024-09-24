import logging
import numpy as np
import matplotlib.pyplot as plt

from util.util_log import setup_logging
from util.util_csv import save_csv
from util.util_image import save_img, save_img_xyz
from util.util_ris_pattern import point_2_phi_pattern, parse_pattern_dbw, phase_2_pattern, phase_2_pattern_xyz
from util.util_analysis_plane import set_theta_phi_loop, get_peaks, get_peak_3rd, get_peak_5th


# ============================================= 主函数 ====================================
# 几何分区法 -- 双波束
def main_multi_scan_2(path_pre,
                      theta1_begin, theta1_over, theta1_step, phi1_begin, phi1_over, phi1_step,
                      theta2_begin, theta2_over, theta2_step, phi2_begin, phi2_over, phi2_step):
    logger.info("main_multi_scan_2:")
    logger.info("theta1_begin=%d, theta1_over=%d, theta1_step=%d, phi1_begin=%d, phi1_over=%d, phi1_step=%d, "
                % (theta1_begin, theta1_over, theta1_step, phi1_begin, phi1_over, phi1_step))
    logger.info("theta2_begin=%d, theta2_over=%d, theta2_step=%d, phi2_begin=%d, phi2_over=%d, phi2_step=%d, "
                % (theta2_begin, theta2_over, theta2_step, phi2_begin, phi2_over, phi2_step))
    # 轮询扫描范围, 完成两个任务: 1.找到最大的副瓣，并保存当时角度, 码阵和方向图; 2.每个角度记录方向图做动图
    theta1_loops = set_theta_phi_loop(theta1_begin, theta1_over, theta1_step)
    phi1_loops = set_theta_phi_loop(phi1_begin, phi1_over, phi1_step)
    theta2_loops = set_theta_phi_loop(theta2_begin, theta2_over, theta2_step)
    phi2_loops = set_theta_phi_loop(phi2_begin, phi2_over, phi2_step)
    i = 0
    # 收敛曲线
    list_third_peak = list()
    # 最大旁瓣结果
    worst_peaks = None
    worst_theta1 = -1
    worst_phi1 = -1
    worst_theta2 = -1
    worst_phi2 = -1
    worst_third_peak = -1000
    worst_third_peak_x = -1
    worst_third_peak_y = -1
    worst_phaseBit_mix = None
    worst_patternBit_mix = None
    worst_patternBit_mix_dbw = None
    worst_patternBit_mix_xyz = None
    worst_patternBit_mix_xyz_x = None
    worst_patternBit_mix_xyz_y = None
    # loop
    for theta1 in theta1_loops:
        for phi1 in phi1_loops:
            for theta2 in theta2_loops:
                for phi2 in phi2_loops:
                    logger.info("--------------------------------------------------------")
                    logger.info("(theta1, phi1)=(%d, %d), (theta2, phi2)=(%d, %d) - %d" % (theta1, phi1, theta2, phi2, i))
                    i = i + 1
                    # 完成两个任务: 1.找到最大的副瓣，并保存当时角度, 码阵和方向图; 2.每个角度记录方向图做动图
                    # 0. 几何分区法
                    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1)
                    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2)
                    # 几何分区法 -- 核心方法
                    rows, cols = phaseBit1.shape
                    phaseBit_mix = np.zeros((rows, cols))
                    phaseBit_mix[:rows // 2, :] = phaseBit1[:rows // 2, :]
                    phaseBit_mix[rows // 2:, :] = phaseBit2[rows // 2:, :]
                    # 计算 phase_mix 的方向图
                    patternBit_mix = phase_2_pattern(phaseBit_mix)
                    patternBit_mix_xyz, x, y, z = phase_2_pattern_xyz(phaseBit_mix)
                    #
                    patternBit_mix_dbw = parse_pattern_dbw(phaseBit_mix)
                    # 1. 每个角度记录方向图做动图
                    path_img_name = "(" + str(theta1) + "," + str(phi1) + "," + str(theta2) + "," + str(phi2) + ")"
                    path_img_patternBit = path_pre + "patternBit_mix/" + path_img_name + ".jpg"
                    save_img(path_img_patternBit, patternBit_mix)
                    path_img_patternBit_xyz = path_pre + "patternBit_mix_xyz/" + path_img_name + ".jpg"
                    save_img_xyz(path_img_patternBit_xyz, np.abs(patternBit_mix_xyz), x, y)
                    path_img_phaseBit = path_pre + "phaseBit_mix/" + path_img_name + ".jpg"
                    save_img(path_img_phaseBit, phaseBit_mix)
                    # 2. 找到最大的副瓣，并保存当时角度, 码阵和方向图
                    peaks = get_peaks(patternBit_mix_dbw)
                    logger.info("peaks.len=%d" % (len(peaks)))
                    if len(peaks) <= 2:
                        logger.info("ATTENTION: no peak!!!!!!!")
                        break
                    logger.info("1st: %f - (%d, %d), 2nd: %f - (%d, %d), 3rd: %f - (%d, %d)"
                                % (peaks[0][0], peaks[0][1][0], peaks[0][1][1],
                                   peaks[1][0], peaks[1][1][0], peaks[1][1][1],
                                   peaks[2][0], peaks[2][1][0], peaks[2][1][1]))
                    third_peak = get_peak_3rd(peaks)
                    if third_peak is None:
                        break
                    third_peak_val = third_peak[0]
                    third_peak_x, third_peak_y = third_peak[1][0], third_peak[1][1]
                    # list_third_peak.append(third_peak_val)
                    peak_record_len = min(6, len(peaks))
                    list_third_peak.append([third_peak_val, third_peak_x, third_peak_y, theta1, phi1, theta2, phi2,
                                            peaks[0][0], peaks[0][1][0], peaks[0][1][1],
                                            peaks[1][0], peaks[1][1][0], peaks[1][1][1],
                                            peaks[2][0], peaks[2][1][0], peaks[2][1][1],
                                            peaks[:peak_record_len]
                                            ])
                    logger.info("third_peak_val: %f - (%d, %d)" % (third_peak_val, third_peak_x, third_peak_y))
                    if worst_third_peak < third_peak_val:
                        worst_peaks = peaks
                        worst_theta1 = theta1
                        worst_phi1 = phi1
                        worst_theta2 = theta2
                        worst_phi2 = phi2
                        worst_phaseBit_mix = phaseBit_mix
                        worst_patternBit_mix = patternBit_mix
                        worst_patternBit_mix_dbw = patternBit_mix_dbw
                        worst_patternBit_mix_xyz = patternBit_mix_xyz
                        worst_patternBit_mix_xyz_x = x
                        worst_patternBit_mix_xyz_y = y
                        worst_third_peak = third_peak_val
                        worst_third_peak_x = third_peak_x
                        worst_third_peak_y = third_peak_y
    # 保存最大旁瓣的结果
    save_img(path_pre + "worst_phaseBit_mix.jpg", worst_phaseBit_mix)
    save_img(path_pre + "worst_patternBit_mix.jpg", worst_patternBit_mix)
    save_img(path_pre + "worst_patternBit_mix_dbw.jpg", worst_patternBit_mix_dbw)
    save_img_xyz(path_pre + "worst_patternBit_mix_xyz.jpg", np.abs(worst_patternBit_mix_xyz),
                 worst_patternBit_mix_xyz_x, worst_patternBit_mix_xyz_y)
    # 写结果
    with open(path_pre + 'worst_third_peak.txt', 'a') as file:
        file.write("worst: (theta1, phi1)=(%f, %f), (theta2, phi2)=(%f, %f)\n"
                   % (worst_theta1, worst_phi1, worst_theta2, worst_phi2))
        file.write("worst_third_peak=%f, worst_third_peak_x=%d, worst_third_peak_y=%d\n"
                   % (worst_third_peak, worst_third_peak_x, worst_third_peak_y))
        peak_record_len = min(6, len(worst_peaks))
        file.write("worst_peaks=%s\n" % worst_peaks[:peak_record_len])
    # 写csv
    save_csv(list_third_peak, path_pre + "list_third_peak.csv")
    save_csv(worst_phaseBit_mix, path_pre + "worst_phaseBit_mix.csv")
    save_csv(worst_patternBit_mix, path_pre + "worst_patternBit_mix.csv")
    save_csv(worst_patternBit_mix_dbw, path_pre + "worst_patternBit_mix_dbw.csv")


# 相位合成法 -- 四波束
def main_multi_scan_4(path_pre,
                      theta1_begin, theta1_over, theta1_step, phi1_begin, phi1_over, phi1_step,
                      theta2_begin, theta2_over, theta2_step, phi2_begin, phi2_over, phi2_step,
                      theta3_begin, theta3_over, theta3_step, phi3_begin, phi3_over, phi3_step,
                      theta4_begin, theta4_over, theta4_step, phi4_begin, phi4_over, phi4_step):
    logger.info("main_multi_scan_4:")
    logger.info("theta1_begin=%d, theta1_over=%d, theta1_step=%d, phi1_begin=%d, phi1_over=%d, phi1_step=%d, "
                % (theta1_begin, theta1_over, theta1_step, phi1_begin, phi1_over, phi1_step))
    logger.info("theta2_begin=%d, theta2_over=%d, theta2_step=%d, phi2_begin=%d, phi2_over=%d, phi2_step=%d, "
                % (theta2_begin, theta2_over, theta2_step, phi2_begin, phi2_over, phi2_step))
    logger.info("theta3_begin=%d, theta3_over=%d, theta3_step=%d, phi3_begin=%d, phi3_over=%d, phi3_step=%d, "
                % (theta3_begin, theta3_over, theta3_step, phi3_begin, phi3_over, phi3_step))
    logger.info("theta4_begin=%d, theta4_over=%d, theta4_step=%d, phi4_begin=%d, phi4_over=%d, phi4_step=%d, "
                % (theta4_begin, theta4_over, theta4_step, phi4_begin, phi4_over, phi4_step))
    # 轮询扫描范围, 完成两个任务: 1.找到最大的副瓣，并保存当时角度, 码阵和方向图; 2.每个角度记录方向图做动图
    theta1_loops = set_theta_phi_loop(theta1_begin, theta1_over, theta1_step)
    phi1_loops = set_theta_phi_loop(phi1_begin, phi1_over, phi1_step)
    theta2_loops = set_theta_phi_loop(theta2_begin, theta2_over, theta2_step)
    phi2_loops = set_theta_phi_loop(phi2_begin, phi2_over, phi2_step)
    theta3_loops = set_theta_phi_loop(theta3_begin, theta3_over, theta3_step)
    phi3_loops = set_theta_phi_loop(phi3_begin, phi3_over, phi3_step)
    theta4_loops = set_theta_phi_loop(theta4_begin, theta4_over, theta4_step)
    phi4_loops = set_theta_phi_loop(phi4_begin, phi4_over, phi4_step)
    i = 0
    # 收敛曲线
    list_peak_5th = list()
    # 最大旁瓣结果
    worst_peaks = None
    worst_theta1 = -1
    worst_phi1 = -1
    worst_theta2 = -1
    worst_phi2 = -1
    worst_peak_5th_val = -1000
    worst_peak_5th_x = -1
    worst_peak_5th_y = -1
    worst_phaseBit_mix = None
    worst_patternBit_mix = None
    worst_patternBit_mix_dbw = None
    worst_patternBit_mix_xyz = None
    worst_patternBit_mix_xyz_x = None
    worst_patternBit_mix_xyz_y = None
    # loop
    for theta1 in theta1_loops:
        for phi1 in phi1_loops:
            for theta2 in theta2_loops:
                for phi2 in phi2_loops:
                    for theta3 in theta3_loops:
                        for phi3 in phi3_loops:
                            for theta4 in theta4_loops:
                                for phi4 in phi4_loops:
                                    logger.info("--------------------------------------------------------")
                                    logger.info("(theta1, phi1)=(%d, %d), (theta2, phi2)=(%d, %d), (theta3, phi3)=(%d, %d), (theta4, phi4)=(%d, %d) - %d"
                                                % (theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4, i))
                                    i = i + 1
                                    # 完成两个任务: 1.找到最大的副瓣，并保存当时角度, 码阵和方向图; 2.每个角度记录方向图做动图
                                    # 0. 相位合成法
                                    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1)
                                    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2)
                                    phase3, phaseBit3, pattern3 = point_2_phi_pattern(theta3, phi3)
                                    phase4, phaseBit4, pattern4 = point_2_phi_pattern(theta4, phi4)
                                    # 几何分区法 -- 核心方法
                                    rows, cols = phaseBit1.shape
                                    phaseBit_mix = np.zeros((rows, cols))
                                    phaseBit_mix[:rows // 2, cols // 2:] = phaseBit1[:rows // 2, cols // 2:]  # 右上半部分
                                    phaseBit_mix[rows // 2:, cols // 2:] = phaseBit2[rows // 2:, cols // 2:]  # 右下半部分
                                    phaseBit_mix[rows // 2:, :cols // 2] = phaseBit3[rows // 2:, :cols // 2]  # 左下半部分
                                    phaseBit_mix[:rows // 2, :cols // 2] = phaseBit4[:rows // 2, :cols // 2]  # 左上半部分
                                    # 计算 phase_mix 的方向图
                                    patternBit_mix = phase_2_pattern(phaseBit_mix)
                                    patternBit_mix_xyz, x, y, z = phase_2_pattern_xyz(phaseBit_mix)
                                    #
                                    patternBit_mix_dbw = parse_pattern_dbw(phaseBit_mix)
                                    # 1. 每个角度记录方向图做动图
                                    path_img_name = "(" + str(theta1) + "," + str(phi1) \
                                                    + "," + str(theta2) + "," + str(phi2) \
                                                    + "," + str(theta3) + "," + str(phi3) \
                                                    + "," + str(theta4) + "," + str(phi4) + ")"
                                    path_img_patternBit = path_pre + "patternBit_mix/" + path_img_name + ".jpg"
                                    save_img(path_img_patternBit, patternBit_mix)
                                    path_img_patternBit_xyz = path_pre + "patternBit_mix_xyz/" + path_img_name + ".jpg"
                                    save_img_xyz(path_img_patternBit_xyz, np.abs(patternBit_mix_xyz), x, y)
                                    path_img_phaseBit = path_pre + "phaseBit_mix/" + path_img_name + ".jpg"
                                    save_img(path_img_phaseBit, phaseBit_mix)
                                    # 2. 找到最大的副瓣，并保存当时角度, 码阵和方向图
                                    peaks = get_peaks(patternBit_mix_dbw)
                                    logger.info("peaks.len=%d" % (len(peaks)))
                                    if len(peaks) <= 2:
                                        logger.info("ATTENTION: no peak!!!!!!!")
                                        break
                                    logger.info("1st: %f - (%d, %d), 2nd: %f - (%d, %d), 3rd: %f - (%d, %d), 4th: %f - (%d, %d), 5th: %f - (%d, %d)"
                                                % (peaks[0][0], peaks[0][1][0], peaks[0][1][1],
                                                   peaks[1][0], peaks[1][1][0], peaks[1][1][1],
                                                   peaks[2][0], peaks[2][1][0], peaks[2][1][1],
                                                   peaks[3][0], peaks[3][1][0], peaks[3][1][1],
                                                   peaks[4][0], peaks[4][1][0], peaks[4][1][1]))
                                    peak_5th = get_peak_5th(peaks)
                                    if peak_5th is None:
                                        break
                                    peak_5th_val = peak_5th[0]
                                    peak_5th_x, peak_5th_y = peak_5th[1][0], peak_5th[1][1]
                                    peak_record_len = min(10, len(peaks))
                                    list_peak_5th.append(
                                        [peak_5th_val, peak_5th_x, peak_5th_y, theta1, phi1, theta2, phi2,
                                         peaks[0][0], peaks[0][1][0], peaks[0][1][1],
                                         peaks[1][0], peaks[1][1][0], peaks[1][1][1],
                                         peaks[2][0], peaks[2][1][0], peaks[2][1][1],
                                         peaks[3][0], peaks[3][1][0], peaks[3][1][1],
                                         peaks[4][0], peaks[4][1][0], peaks[4][1][1],
                                         peaks[:peak_record_len]
                                         ])
                                    logger.info(
                                        "peak_val_5th: %f - (%d, %d)" % (peak_5th_val, peak_5th_x, peak_5th_y))
                                    if worst_peak_5th_val < peak_5th_val:
                                        worst_peaks = peaks
                                        worst_theta1 = theta1
                                        worst_phi1 = phi1
                                        worst_theta2 = theta2
                                        worst_phi2 = phi2
                                        worst_phaseBit_mix = phaseBit_mix
                                        worst_patternBit_mix = patternBit_mix
                                        worst_patternBit_mix_dbw = patternBit_mix_dbw
                                        worst_patternBit_mix_xyz = patternBit_mix_xyz
                                        worst_patternBit_mix_xyz_x = x
                                        worst_patternBit_mix_xyz_y = y
                                        worst_peak_5th_val = peak_5th_val
                                        worst_peak_5th_x = peak_5th_x
                                        worst_peak_5th_y = peak_5th_y
    # 保存最大旁瓣的结果
    save_img(path_pre + "worst_phaseBit_mix.jpg", worst_phaseBit_mix)
    save_img(path_pre + "worst_patternBit_mix.jpg", worst_patternBit_mix)
    save_img(path_pre + "worst_patternBit_mix_dbw.jpg", worst_patternBit_mix_dbw)
    save_img_xyz(path_pre + "worst_patternBit_mix_xyz.jpg", np.abs(worst_patternBit_mix_xyz),
                 worst_patternBit_mix_xyz_x, worst_patternBit_mix_xyz_y)
    #
    with open(path_pre + 'worst_peak_5th.txt', 'a') as file:
        file.write("worst: (theta1, phi1)=(%f, %f), (theta2, phi2)=(%f, %f)\n"
                   % (worst_theta1, worst_phi1, worst_theta2, worst_phi2))
        file.write("worst_peak_5th_val=%f, worst_peak_5th_x=%d, worst_peak_5th_y=%d\n"
                   % (worst_peak_5th_val, worst_peak_5th_x, worst_peak_5th_y))
        peak_record_len = min(10, len(worst_peaks))
        file.write("worst_peaks=%s\n" % worst_peaks[:peak_record_len])
    #
    save_csv(list_peak_5th, path_pre + "list_peak_5th.csv")
    save_csv(worst_phaseBit_mix, path_pre + "worst_phaseBit_mix.csv")
    save_csv(worst_patternBit_mix, path_pre + "worst_patternBit_mix.csv")
    save_csv(worst_patternBit_mix_dbw, path_pre + "worst_patternBit_mix_dbw.csv")




if __name__ == '__main__':
    # 配置日志，默认打印到控制台，也可以设置打印到文件
    setup_logging()
    # setup_logging(log_file="../../files/logs/log_multi_beam_PS.log")

    # 获取日志记录器并记录日志
    logger = logging.getLogger("[RIS-multi-beam-PS-1bit]")
    logger.info("1bit-RIS-multi-beam-NN: Geometric partitioning method")
    # 波束扫描 -- 几何分区法: 主函数
    # main_multi_scan_2("../../files/multi-beam-scan/NN/2-(20,40,10,-30,30,10,20,40,10,150,210,10)/",
    #                   20, 40, 10, -30, 30, 10, 20, 40, 10, 150, 210, 10)
    # main_multi_scan_2("../../files/multi-beam-scan/NN/2-(30,30,0,-5,5,15,30,30,0,175,185,15)/",
    #                   30, 30, 0, -5, -5, 15, 30, 30, 0, 175, 185, 15)
    main_multi_scan_4("../../files/multi-beam-scan/NN/4-(30,30,0,-5,5,15,30,30,0,85,95,15,30,30,0,175,185,15,30,30,0,265,275,15)/",
                      30,30,0,-5,-5,15,30,30,0,85,85,15,30,30,0,175,175,15,30,30,0,265,275,15)
    # main_multi_scan_4("../../files/multi-beam-scan/PS/4-(30,30,0,-20,20,10,30,30,0,70,110,10,30,30,0,160,200,10,30,30,0,250,290,10)/",
    #                   30, 30, 0, -20, 20, 10, 30, 30, 0, 70, 110, 10, 30, 30, 0, 160, 200, 10, 30, 30, 0, 250, 290, 10)