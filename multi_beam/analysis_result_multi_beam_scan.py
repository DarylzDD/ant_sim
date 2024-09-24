import csv
import logging
import numpy as np
import imageio.v3 as iio
import os
import matplotlib.pyplot as plt
import pandas as pd

from util.util_log import setup_logging
from util.util_ris_pattern import parse_pattern_dbw, phase_2_pattern


# 读取CSV文件的第一列数据
def read_csv_peak(path_csv):
    # 读取CSV文件
    data = pd.read_csv(path_csv)
    # 获取第一列的数据
    first_column_data = data.iloc[:, 0]  # 使用iloc获取第一列的数据
    return first_column_data


# 绘制折线图
def plot_first_column_and_get_max(datas):
    plt.figure(figsize=(10, 5))
    plt.plot(datas, marker='o')
    # plt.title('Line Chart of First Column')
    plt.xlabel('Index')
    plt.ylabel('PSL max (dB)')
    plt.grid(True)
    plt.show()


# 波束方向图生成动图
def create_gif(image_folder, output_file, duration):
    # 获取文件夹下所有的jpg文件
    images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    # 按照文件名排序
    images.sort()
    # 读取所有图片
    frames = [iio.imread(os.path.join(image_folder, img)) for img in images]
    # 写入GIF文件
    iio.imwrite(output_file, frames, extension=".gif", loop=0, fps=1/duration)


# 波束方向图生成动图
def multi_beam_scan_gif():
    # 使用方法
    create_gif("../../files/multi-beam-scan/PS/4-(30,30,0,-20,20,10,30,30,0,70,110,10,30,30,0,160,200,10,30,30,0,250,290,10)/patternBit_mix_xyz",
               "../../files/multi-beam-scan/PS/4-(30,30,0,-20,20,10,30,30,0,70,110,10,30,30,0,160,200,10,30,30,0,250,290,10)/patternBit_mix_xyz.gif",
               0.2)  # duration 是每帧显示的时间，单位是秒

#
def multi_beam_scan_peak_img():
    path_csv = "../../files/multi-beam-scan/PS/4-(30,30,0,-20,20,10,30,30,0,70,110,10,30,30,0,160,200,10,30,30,0,250,290,10)/list_peak_5th.csv"
    first_column_data = read_csv_peak(path_csv)
    # 返回第一列的最大值
    max_value = first_column_data.max()
    print("max_value:", max_value)
    plot_first_column_and_get_max(first_column_data)


if __name__ == '__main__':
    # 配置日志，默认打印到控制台，也可以设置打印到文件
    setup_logging()
    # setup_logging(log_file="../../files/logs/log_analysis_multi_beam.log")
    # 获取日志记录器并记录日志
    logger = logging.getLogger("[analysis][RIS-multi-beam]")
    logger.info("[analysis][RIS-multi-beam]: analysis")

    # 生成动图
    multi_beam_scan_gif()
    # multi_beam_scan_peak_img()