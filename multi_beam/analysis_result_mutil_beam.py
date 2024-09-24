import csv
import logging
import numpy as np
import matplotlib.pyplot as plt

from util.util_log import setup_logging
from util.util_ris_pattern import parse_pattern_dbw, phase_2_pattern, eps
from util.util_csv import read_csv_to_numpy_array
from util.util_analysis_plane import get_peaks, get_peak_3rd, get_peak_5th


# =================================================== csv相关 =====================================================
# 读取csv码阵
def read_csv_to_2d_array_with_numbers(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 尝试将每列转换为浮点数，如果失败则保持为字符串
            processed_row = [float(element) if element.replace('.', '', 1).isdigit() else element for element in row]
            data.append(processed_row)
    return data


# =================================================== 画图相关 =====================================================
# 画三维图
def draw_3d(data, title):
    # 创建一个图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 生成x和y坐标
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)
    # 绘制三维图
    ax.plot_surface(x, y, data, cmap='viridis')
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('normalized pattern')
    ax.set_title(title)
    # 显示图形
    plt.show()


# 画曲线图
def draw_curve(arr_NN, arr_PS, arr_CNN, xlabel, xlim, xticks, ylabel, title):
    # 创建一个图形对象
    plt.figure()
    # 绘制曲线图
    plt.plot(arr_NN, label="NN", color='black')
    plt.plot(arr_PS, label="PS", color='blue')
    # plt.plot(arr_GA_AP, label="GA-AP", color='green')
    plt.plot(arr_CNN, label="CNN", color='purple')
    # 设置标签和标题
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    # plt.xticks(xticks)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # 显示图形
    plt.show()


# 画曲线图
def draw_curve_PS(arr_PS, xlabel, xlim, xticks, ylabel, title):
    # 创建一个图形对象
    plt.figure()
    # 绘制曲线图
    # plt.plot(arr_NN, label="NN", color='black')
    plt.plot(arr_PS, label="PS", color='blue')
    # plt.plot(arr_GA_AP, label="GA-AP", color='green')
    # plt.plot(arr_CNN, label="CNN", color='purple')
    # 设置标签和标题
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    # plt.xticks(xticks)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # 显示图形
    plt.show()


# 画曲线图
def draw_curve_bak(arr_NN, arr_PS, arr_GA_AP, arr_CNN, xlabel, xlim, xticks, ylabel, title):
    # 创建一个图形对象
    plt.figure()
    # 绘制曲线图
    plt.plot(arr_NN, label="NN", color='black')
    plt.plot(arr_PS, label="PS", color='blue')
    plt.plot(arr_GA_AP, label="GA-AP", color='green')
    plt.plot(arr_CNN, label="CNN", color='purple')
    # 设置标签和标题
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    # plt.xticks(xticks)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # 显示图形
    plt.show()


# =================================================== 正式代码 =====================================================
# 寻找最大值的行和列
def find_max_col_row(data):
    # 示例二维数组
    # data = np.array([[1.1, 2.2, 3.3],
    #                  [4.4, 5.5, 6.6],
    #                  [7.7, 8.8, 9.9]])
    # 找到最大元素的扁平化索引
    flat_index = np.argmax(data)
    # 将扁平化索引转换为二维数组的行和列
    row, col = np.unravel_index(flat_index, data.shape)
    # 输出最大元素所在行的所有元素列表
    row_elements = data[row, :].tolist()
    # 输出最大元素所在列的所有元素列表
    col_elements = data[:, col].tolist()
    # 输出结果
    logger.info(f"最大元素位于第 {row + 1} 行，该行的所有元素为: {row_elements}")
    logger.info(f"最大元素位于第 {col + 1} 列，该列的所有元素为: {col_elements}")
    return col_elements, row_elements


# 比较波束图
def multi_beam_analysis():
    # 读取布阵结果
    # path_NN_phaseBit_mix = "../../files/multi-beam/NN/2-(30,0,30,180)/phase_mix_lr.csv"
    # path_PS_phaseBit_mix = "../../files/multi-beam/PS/2-(30,0,30,180)/phaseBit_mix.csv"
    # path_PS_phase_mix = "../../files/multi-beam/PS/2-(30,0,30,180)/phase_mix.csv"
    # # path_GA_AP_phaseBit_mix = "../../files/multi-beam/GA-AP-10-30/2-(30,0,30,180)/phaseBit_mix.csv"
    # path_CNN_phaseBit_mix = "../../files/multi-beam/CNN/2-(30,0,30,180)/CNN1-50-16/phaseBit_mix.csv"
    path_NN_phaseBit_mix = "../../files/multi-beam/NN/4-(30,0,30,90,30,180,30,270)/phase_mix.csv"
    path_PS_phaseBit_mix = "../../files/multi-beam/PS/4-(30,0,30,90,30,180,30,270)/phaseBit_mix.csv"
    path_PS_phase_mix = "../../files/multi-beam/PS/4-(30,0,30,90,30,180,30,270)/phase_mix.csv"
    path_CNN_phaseBit_mix = "../../files/multi-beam/CNN/4-(30,0,30,90,30,180,30,270)/CNN1-50-16/phaseBit_mix.csv"
    # path_NN_phaseBit_mix = "../../files/multi-beam/NN/4-(30,0,30,60,30,120,30,180)/phase_mix.csv"
    # path_PS_phaseBit_mix = "../../files/multi-beam/PS/4-(30,0,30,60,30,120,30,180)/phaseBit_mix.csv"
    # path_PS_phase_mix = "../../files/multi-beam/PS/4-(30,0,30,60,30,120,30,180)/phase_mix.csv"
    # path_CNN_phaseBit_mix = "../../files/multi-beam/CNN/4-(30,0,30,60,30,120,30,180)/CNN1-50-16/phaseBit_mix.csv"
    NN_phaseBit_mix = read_csv_to_2d_array_with_numbers(path_NN_phaseBit_mix)
    PS_phaseBit_mix = read_csv_to_2d_array_with_numbers(path_PS_phaseBit_mix)
    PS_phase_mix = read_csv_to_2d_array_with_numbers(path_PS_phase_mix)
    # GA_AP_phaseBit_mix = read_csv_to_2d_array_with_numbers(path_GA_AP_phaseBit_mix)
    CNN_phaseBit_mix = read_csv_to_2d_array_with_numbers(path_CNN_phaseBit_mix)
    # 转np.array
    logger.info("read_csv_to_2d_array_with_numbers: %s, %s, %s"
                % (path_NN_phaseBit_mix, path_PS_phaseBit_mix, path_PS_phase_mix))
    np_NN_phaseBit_mix = np.array(NN_phaseBit_mix)
    np_PS_phaseBit_mix = np.array(PS_phaseBit_mix)
    np_PS_phase_mix = np.array(PS_phase_mix)
    # np_GA_AP_phaseBit_mix = np.array(GA_AP_phaseBit_mix)
    np_CNN_phaseBit_mix = np.array(CNN_phaseBit_mix)
    # 计算方向图
    pattern_dbw_NN = parse_pattern_dbw(np_NN_phaseBit_mix)
    pattern_dbw_PS = parse_pattern_dbw(np_PS_phaseBit_mix)
    # pattern_dbw_GA_AP = parse_pattern_dbw(np_GA_AP_phaseBit_mix)
    pattern_dbw_CNN = parse_pattern_dbw(np_CNN_phaseBit_mix)
    # 画3d图
    draw_3d(pattern_dbw_NN, "normalized pattern -- NN")
    draw_3d(pattern_dbw_PS, "normalized pattern -- PS")
    # draw_3d(pattern_dbw_GA_AP, "normalized pattern -- GA-AP")
    draw_3d(pattern_dbw_CNN, "normalized pattern -- CNN")
    # 找到最大行列
    max_col_NN, max_row_NN = find_max_col_row(pattern_dbw_NN)
    max_col_PS, max_row_PS = find_max_col_row(pattern_dbw_PS)
    # max_col_GA_AP, max_row_GA_AP = find_max_col_row(pattern_dbw_GA_AP)
    max_col_CNN, max_row_CNN = find_max_col_row(pattern_dbw_CNN)
    # draw_curve(max_col_NN, max_col_PS, max_col_GA_AP, max_col_CNN,
    #            "degree", [0, 360], [0, 60, 120, 180, 240, 300, 360],
    #            "gain", "normalized pattern (phi)")  # 横坐标轴是对的[0, 360]
    # draw_curve(max_row_NN, max_row_PS, max_row_GA_AP, max_row_CNN,
    #            "degree", [0, 360], [0, 15, 30, 45, 60, 75, 90],
    #            "gain", "normalized pattern (theta)")  # 横坐标轴不对[0, 90]
    draw_curve(max_col_NN, max_col_PS, max_col_CNN,
               "degree", [0, 360], [0, 60, 120, 180, 240, 300, 360],
               "gain", "normalized pattern (phi)")  # 横坐标轴是对的[0, 360]
    draw_curve(max_row_NN, max_row_PS, max_row_CNN,
               "degree", [0, 360], [0, 15, 30, 45, 60, 75, 90],
               "gain", "normalized pattern (theta)")  # 横坐标轴不对[0, 90]
    #
    #
    #
    # pattern_NN = phase_2_pattern(np_NN_phaseBit_mix)
    # pattern_PS = phase_2_pattern(np_PS_phaseBit_mix)
    # # pattern_GA_AP = phase_2_pattern(np_GA_AP_phaseBit_mix)
    # pattern_CNN = phase_2_pattern(np_CNN_phaseBit_mix)
    # max_col_NN_2, max_row_NN_2 = find_max_col_row(pattern_NN)
    # max_col_PS_2, max_row_PS_2 = find_max_col_row(pattern_PS)
    # # max_col_GA_AP_2, max_row_GA_AP_2 = find_max_col_row(pattern_GA_AP)
    # max_col_CNN_2, max_row_CNN_2 = find_max_col_row(pattern_CNN)
    # draw_curve(max_col_NN_2, max_col_PS_2, max_col_CNN_2,
    #            "degree", [0, 360], [0, 60, 120, 180, 240, 300, 360],
    #            "gain", "normalized pattern (phi)")  # 横坐标轴是对的[0, 360]
    # draw_curve(max_row_NN_2, max_row_PS_2, max_row_CNN_2,
    #            "degree", [0, 360], [0, 15, 30, 45, 60, 75, 90],
    #            "gain", "normalized pattern (theta)")  # 横坐标轴不对[0, 90]
    # # draw_curve_bak(max_col_NN_2, max_col_PS_2, max_col_GA_AP_2, max_col_CNN_2,
    # #            "degree", [0, 360], [0, 60, 120, 180, 240, 300, 360],
    # #            "gain", "normalized pattern (phi)")  # 横坐标轴是对的[0, 360]
    # # draw_curve_bak(max_row_NN_2, max_row_PS_2, max_row_GA_AP_2, max_row_CNN_2,
    # #            "degree", [0, 360], [0, 15, 30, 45, 60, 75, 90],
    # #            "gain", "normalized pattern (theta)")  # 横坐标轴不对[0, 90]


# 比较波束图
def multi_beam_analysis_tmp():
    # 读取布阵结果
    # path_NN_phaseBit_mix = "../../files/multi-beam/NN/2-(30,0,30,180)/phase_mix_lr.csv"
    # path_PS_phaseBit_mix = "../../files/multi-beam/PS/2-(30,0,30,180)/phaseBit_mix.csv"
    # path_PS_phase_mix = "../../files/multi-beam/PS/2-(30,0,30,180)/phase_mix.csv"
    # # path_GA_AP_phaseBit_mix = "../../files/multi-beam/GA-AP-10-30/2-(30,0,30,180)/phaseBit_mix.csv"
    # path_CNN_phaseBit_mix = "../../files/multi-beam/CNN/2-(30,0,30,180)/CNN1-50-16/phaseBit_mix.csv"
    # path_NN_phaseBit_mix = "../../files/multi-beam/NN/4-(30,0,30,90,30,180,30,270)/phase_mix.csv"
    path_PS_phaseBit_mix = "../../files/multi-beam/PS/B-32-uni-(30,11.25step)/phaseBit_mix.csv"
    path_PS_phase_mix = "../../files/multi-beam/PS/B-32-uni-(30,11.25step)/phase_mix.csv"
    # path_CNN_phaseBit_mix = "../../files/multi-beam/CNN/4-(30,0,30,90,30,180,30,270)/CNN1-50-16/phaseBit_mix.csv"
    # path_NN_phaseBit_mix = "../../files/multi-beam/NN/4-(30,0,30,60,30,120,30,180)/phase_mix.csv"
    # path_PS_phaseBit_mix = "../../files/multi-beam/PS/4-(30,0,30,60,30,120,30,180)/phaseBit_mix.csv"
    # path_PS_phase_mix = "../../files/multi-beam/PS/4-(30,0,30,60,30,120,30,180)/phase_mix.csv"
    # path_CNN_phaseBit_mix = "../../files/multi-beam/CNN/4-(30,0,30,60,30,120,30,180)/CNN1-50-16/phaseBit_mix.csv"
    # NN_phaseBit_mix = read_csv_to_2d_array_with_numbers(path_NN_phaseBit_mix)
    PS_phaseBit_mix = read_csv_to_2d_array_with_numbers(path_PS_phaseBit_mix)
    PS_phase_mix = read_csv_to_2d_array_with_numbers(path_PS_phase_mix)
    # GA_AP_phaseBit_mix = read_csv_to_2d_array_with_numbers(path_GA_AP_phaseBit_mix)
    # CNN_phaseBit_mix = read_csv_to_2d_array_with_numbers(path_CNN_phaseBit_mix)
    # 转np.array
    # logger.info("read_csv_to_2d_array_with_numbers: %s, %s, %s"
    #             % (path_NN_phaseBit_mix, path_PS_phaseBit_mix, path_PS_phase_mix))
    # np_NN_phaseBit_mix = np.array(NN_phaseBit_mix)
    np_PS_phaseBit_mix = np.array(PS_phaseBit_mix)
    np_PS_phase_mix = np.array(PS_phase_mix)
    # np_GA_AP_phaseBit_mix = np.array(GA_AP_phaseBit_mix)
    # np_CNN_phaseBit_mix = np.array(CNN_phaseBit_mix)
    # 计算方向图
    # pattern_dbw_NN = parse_pattern_dbw(np_NN_phaseBit_mix)
    pattern_dbw_PS = parse_pattern_dbw(np_PS_phaseBit_mix)
    # pattern_dbw_GA_AP = parse_pattern_dbw(np_GA_AP_phaseBit_mix)
    # pattern_dbw_CNN = parse_pattern_dbw(np_CNN_phaseBit_mix)
    # 画3d图
    # draw_3d(pattern_dbw_NN, "normalized pattern -- NN")
    draw_3d(pattern_dbw_PS, "normalized pattern -- PS")
    # draw_3d(pattern_dbw_GA_AP, "normalized pattern -- GA-AP")
    # draw_3d(pattern_dbw_CNN, "normalized pattern -- CNN")
    # 找到最大行列
    # max_col_NN, max_row_NN = find_max_col_row(pattern_dbw_NN)
    max_col_PS, max_row_PS = find_max_col_row(pattern_dbw_PS)
    # max_col_GA_AP, max_row_GA_AP = find_max_col_row(pattern_dbw_GA_AP)
    # max_col_CNN, max_row_CNN = find_max_col_row(pattern_dbw_CNN)
    # draw_curve(max_col_NN, max_col_PS, max_col_GA_AP, max_col_CNN,
    #            "degree", [0, 360], [0, 60, 120, 180, 240, 300, 360],
    #            "gain", "normalized pattern (phi)")  # 横坐标轴是对的[0, 360]
    # draw_curve(max_row_NN, max_row_PS, max_row_GA_AP, max_row_CNN,
    #            "degree", [0, 360], [0, 15, 30, 45, 60, 75, 90],
    #            "gain", "normalized pattern (theta)")  # 横坐标轴不对[0, 90]
    draw_curve_PS(max_col_PS,
               "degree", [0, 360], [0, 60, 120, 180, 240, 300, 360],
               "gain", "normalized pattern (phi)")  # 横坐标轴是对的[0, 360]
    draw_curve_PS(max_row_PS,
               "degree", [0, 360], [0, 15, 30, 45, 60, 75, 90],
               "gain", "normalized pattern (theta)")  # 横坐标轴不对[0, 90]
    #
    #
    #
    # pattern_NN = phase_2_pattern(np_NN_phaseBit_mix)
    pattern_PS = phase_2_pattern(np_PS_phaseBit_mix)
    # # pattern_GA_AP = phase_2_pattern(np_GA_AP_phaseBit_mix)
    # pattern_CNN = phase_2_pattern(np_CNN_phaseBit_mix)
    # max_col_NN_2, max_row_NN_2 = find_max_col_row(pattern_NN)
    max_col_PS_2, max_row_PS_2 = find_max_col_row(pattern_PS)
    # # max_col_GA_AP_2, max_row_GA_AP_2 = find_max_col_row(pattern_GA_AP)
    # max_col_CNN_2, max_row_CNN_2 = find_max_col_row(pattern_CNN)
    draw_curve_PS(max_col_PS_2,
               "degree", [0, 360], [0, 60, 120, 180, 240, 300, 360],
               "gain", "pattern (phi)")  # 横坐标轴是对的[0, 360]
    draw_curve_PS(max_row_PS_2,
               "degree", [0, 360], [0, 15, 30, 45, 60, 75, 90],
               "gain", "pattern (theta)")  # 横坐标轴不对[0, 90]
    # # draw_curve_bak(max_col_NN_2, max_col_PS_2, max_col_GA_AP_2, max_col_CNN_2,
    # #            "degree", [0, 360], [0, 60, 120, 180, 240, 300, 360],
    # #            "gain", "normalized pattern (phi)")  # 横坐标轴是对的[0, 360]
    # # draw_curve_bak(max_row_NN_2, max_row_PS_2, max_row_GA_AP_2, max_row_CNN_2,
    # #            "degree", [0, 360], [0, 15, 30, 45, 60, 75, 90],
    # #            "gain", "normalized pattern (theta)")  # 横坐标轴不对[0, 90]


# 绘制折线图
def plot_first_column_and_get_max(datas, title):
    plt.figure(figsize=(10, 5))
    plt.plot(datas, marker='o')
    plt.title(title)
    plt.xlabel('iterations')
    plt.ylabel('PSL max (dB)')
    plt.grid(True)
    plt.show()


def multi_beam_random_psll():
    list_psll_nn_2_0_90 = [-12.153437774979787, -12.52575965461449, -12.19228233251728, -12.117221786900675,
                    -12.415072406830545, -11.905765099133337, -12.130812869182641, -11.99270844044764,
                    -12.133008112061939, -12.010217505738218, -12.48155390044823, -11.964593918508452,
                    -11.911279184472141, -12.021210987845347, -12.273918470175602, -12.075250616852959,
                    -12.16948857602028, -12.506723173313697, -12.26872244551931, -12.054144197362369,
                    -11.960028781371875, -12.331231177047927, -12.589933330519521, -11.790858879301808,
                    -11.984391588079014, -12.1595153037206, -11.867748346954519, -12.395190523442732,
                    -12.39944736119953, -12.359151001814787]
    plot_first_column_and_get_max(list_psll_nn_2_0_90, "NN-(30,0,30,90)")
    list_psll_nn_2_0_180 = [-11.1727895320235, -11.202385682333752, -11.140729082175016, -11.38011299634632,
                           -11.304939740501254, -11.71143974813797, -11.19051607500105, -11.293619697791158,
                           -11.191234705009593, -11.46853317439469, -11.426328114236071, -11.14210716877395,
                           -11.080676361855431, -10.977902885360724, -11.088641924434796, -11.00874916710181,
                           -11.760994936553502, -11.199065233140741, -11.259948089430832, -11.353994783208247,
                           -11.799596330647548, -11.421697758753727, -11.265977651529331, -11.152195377291713,
                           -10.94036942578985, -11.032068746963583, -10.985617417656822, -11.02054877173227,
                           -11.330534462457544, -11.356169841991115]
    plot_first_column_and_get_max(list_psll_nn_2_0_180, "NN-(30,0,30,180)")
    #
    list_psll_ps_2_0_90 = [-12.296197175363972, -12.06302306285405, -12.013487516351532, -12.155492187202873, -12.041386023599461,
                 -11.98902147919856, -11.940165386884718, -12.331641327093694, -12.156551527009912, -11.969508509350922,
                 -11.986130396801594, -11.955125527947803, -12.267239791788674, -11.979474166304085,
                 -12.035842244567842, -12.288260259213699, -12.406811630028942, -12.158949897904572,
                 -11.783016129118131, -12.203648241508771, -12.050667917830333, -11.93539794425478, -12.174048087653883,
                 -11.860330507618135, -12.021829547223016, -11.715622632587614, -12.073334338920088,
                 -12.132856838666363, -12.175723644922483, -12.08669823961351]
    plot_first_column_and_get_max(list_psll_ps_2_0_90, "PS-(30,0,30,90)")
    list_psll_ps_2_0_180 = [-12.4041082108455, -12.176995504102276, -12.242020109910527, -12.478113555745136, -12.473894651424493,
                 -12.425019813498368, -12.368195073981864, -12.444016556036138, -12.272260693938637,
                 -12.412414804728215, -12.55586078042207, -12.303404332395269, -12.602417293516282, -12.232733976246386,
                 -12.255983259683562, -12.3354159412191, -12.315985637486161, -12.606789840396004, -12.469829587150967,
                 -12.270161286224697, -12.050208242345448, -12.342157764217937, -12.25601636338876, -12.506278325820952,
                 -12.300184779662937, -12.408199551082165, -12.423897701856099, -12.23739954023812, -12.219655395795284,
                 -12.625474281109831]
    plot_first_column_and_get_max(list_psll_ps_2_0_180, "PS-(30,0,30,180)")


def multi_beam_psll_peaks(phase):
    pattern = phase_2_pattern(phase)
    pattern_dbw = 20 * np.log10(np.abs(pattern) / np.max(np.max(np.abs(pattern))) + eps)
    peaks = get_peaks(pattern_dbw)
    return peaks


def multi_beam_psll_item(name, path_csv):
    phase = read_csv_to_numpy_array(path_csv)
    peak = multi_beam_psll_peaks(phase)
    logger.info("[%s] 1st: %4f(%d,%d), 2nd: %4f(%d,%d), 3rd: %4f(%d,%d), 4th: %4f(%d,%d), 5th: %4f(%d,%d), 6th: %4f(%d,%d)" % (
        name,
        peak[0][0], peak[0][1][0], peak[0][1][1]/4,
        peak[1][0], peak[1][1][0], peak[1][1][1]/4,
        peak[2][0], peak[2][1][0], peak[2][1][1]/4,
        peak[3][0], peak[3][1][0], peak[3][1][1]/4,
        peak[4][0], peak[4][1][0], peak[4][1][1]/4,
        peak[5][0], peak[5][1][0], peak[5][1][1]/4))
    # logger.info(
    #     "[%s] 1st: %4f(%d,%d), 2nd: %4f(%d,%d), 3rd: %4f(%d,%d), 4th: %4f(%d,%d), 5th: %4f(%d,%d), "
    #     "6th: %4f(%d,%d), 7th: %4f(%d,%d), 8th: %4f(%d,%d), 9th: %4f(%d,%d), 10th: %4f(%d,%d),"
    #     "11th: %4f(%d,%d), 12th: %4f(%d,%d), 13th: %4f(%d,%d), 14th: %4f(%d,%d), 15th: %4f(%d,%d), "
    #     "16th: %4f(%d,%d), 17th: %4f(%d,%d), 18th: %4f(%d,%d), 19th: %4f(%d,%d), 20th: %4f(%d,%d)" % (
    #         name,
    #         peak[0][0], peak[0][1][0], peak[0][1][1] / 4,
    #         peak[1][0], peak[1][1][0], peak[1][1][1] / 4,
    #         peak[2][0], peak[2][1][0], peak[2][1][1] / 4,
    #         peak[3][0], peak[3][1][0], peak[3][1][1] / 4,
    #         peak[4][0], peak[4][1][0], peak[4][1][1] / 4,
    #         peak[5][0], peak[5][1][0], peak[5][1][1] / 4,
    #         peak[6][0], peak[6][1][0], peak[6][1][1] / 4,
    #         peak[7][0], peak[7][1][0], peak[7][1][1] / 4,
    #         peak[8][0], peak[8][1][0], peak[8][1][1] / 4,
    #         peak[9][0], peak[9][1][0], peak[9][1][1] / 4,
    #         peak[10][0], peak[10][1][0], peak[10][1][1] / 4,
    #         peak[11][0], peak[11][1][0], peak[11][1][1] / 4,
    #         peak[12][0], peak[12][1][0], peak[12][1][1] / 4,
    #         peak[13][0], peak[13][1][0], peak[13][1][1] / 4,
    #         peak[14][0], peak[14][1][0], peak[14][1][1] / 4,
    #         peak[15][0], peak[15][1][0], peak[15][1][1] / 4,
    #         peak[16][0], peak[16][1][0], peak[16][1][1] / 4,
    #         peak[17][0], peak[17][1][0], peak[17][1][1] / 4,
    #         peak[18][0], peak[18][1][0], peak[18][1][1] / 4,
    #         peak[19][0], peak[19][1][0], peak[19][1][1] / 4))


def multi_beam_psll():
    # multi_beam_psll_item("NN-2-(30,0,30,90)", "../../files/multi-beam/2bit/NN/2-(30,0,30,90)/phaseBit_mix_ud.csv")
    # multi_beam_psll_item("NN-2-(30,0,30,180)", "../../files/multi-beam/2bit/NN/2-(30,0,30,180)/phaseBit_mix_ud.csv")
    # multi_beam_psll_item("NN-4-(30,0,30,60,30,120,30,180)",
    #                      "../../files/multi-beam/2bit/NN/4-(30,0,30,60,30,120,30,180)/phase_mix.csv")
    # multi_beam_psll_item("NN-4-(30,0,30,90,30,180,30,270)",
    #                      "../../files/multi-beam/2bit/NN/4-(30,0,30,90,30,180,30,270)/phase_mix.csv")
    # #
    # multi_beam_psll_item("PS-2-(30,0,30,90)", "../../files/multi-beam/2bit/PS/2-(30,0,30,90)/phaseBit_mix.csv")
    # multi_beam_psll_item("PS-2-(30,0,30,180)", "../../files/multi-beam/2bit/PS/2-(30,0,30,180)/phaseBit_mix.csv")
    # multi_beam_psll_item("PS-4-(30,0,30,60,30,120,30,180)",
    #                      "../../files/multi-beam/2bit/PS/4-(30,0,30,60,30,120,30,180)/phaseBit_mix.csv")
    # multi_beam_psll_item("PS-4-(30,0,30,90,30,180,30,270)",
    #                      "../../files/multi-beam/2bit/PS/4-(30,0,30,90,30,180,30,270)/phaseBit_mix.csv")
    #
    #
    # multi_beam_psll_item("NN-8-cub", "../../files/multi-beam/1bit/NN/8-cub-(30,45step)/phaseBit_mix.csv")
    # multi_beam_psll_item("NN-8-ang", "../../files/multi-beam/1bit/NN/8-ang-(30,45step)/phaseBit_mix.csv")
    # multi_beam_psll_item("NN-8-spa", "../../files/multi-beam/1bit/NN/B-8-uni-(30,45step)/phase_mix.csv")
    #
    #
    #
    multi_beam_psll_item("NN-PS-2-(30,0,30,90)", "../../files/multi-beam/1bit/NN-PS/2-(30,0,30,90)/phaseBit_mix.csv")
    multi_beam_psll_item("NN-PS-2-(30,0,30,180)", "../../files/multi-beam/1bit/NN-PS/2-(30,0,30,180)/phaseBit_mix.csv")
    multi_beam_psll_item("NN-PS-4-(30,0,30,60,30,120,30,180)",
                         "../../files/multi-beam/1bit/NN-PS/4-(30,0,30,60,30,120,30,180)/phase_mix.csv")
    multi_beam_psll_item("NN-PS-4-(30,0,30,90,30,180,30,270)",
                         "../../files/multi-beam/1bit/NN-PS/4-(30,0,30,90,30,180,30,270)/phase_mix.csv")



if __name__ == '__main__':
    # 配置日志，默认打印到控制台，也可以设置打印到文件
    setup_logging()
    # setup_logging(log_file="../../files/logs/log_analysis_multi_beam.log")
    # 获取日志记录器并记录日志
    logger = logging.getLogger("[analysis][RIS-multi-beam]")
    logger.info("[analysis][RIS-multi-beam]: analysis")

    # 主函数
    # multi_beam_analysis_tmp()
    # multi_beam_random_psll()
    multi_beam_psll()