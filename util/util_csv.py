import csv
import numpy as np

# ============================================= excel相关 =========================================================
def save_csv(data, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    # logger.info("数据已成功保存到CSV文件: %s" % (file_path))


def read_csv_to_numpy_array(file_path):
    """
    读取一个.csv文件并将其转换为numpy的二维数组。

    :param file_path: .csv文件的路径
    :return: numpy的二维数组
    """
    try:
        # 使用numpy的loadtxt函数读取csv文件
        data = np.loadtxt(file_path, delimiter=',')  # 使用制表符作为分隔符
        return data
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

