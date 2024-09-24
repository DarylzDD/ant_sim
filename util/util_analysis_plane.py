import numpy as np

# theta 和 phi 轮询范围设置
def set_theta_phi_loop(begin, over, step):
    if begin == over:
        loop_values = [begin]
    else:
        loop_values = range(begin, over + step, step)
    return loop_values


# 寻找arr中的峰
def get_peaks(arr):
    # 获取数组的维度
    rows, cols = arr.shape
    # 用一个列表来存储每个峰的高度及其坐标
    peaks = []
    for i in range(rows):
        for j in range(cols):
            # 获取当前元素的高度
            current_height = arr[i, j]
            neighbors = []
            # 获取所有邻居（包括四个对角方向的邻居）
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = (i + di) % rows
                    nj = (j + dj) % cols
                    neighbors.append(arr[ni, nj])
            # 判断当前元素是否大于周围的所有元素
            if all(current_height > neighbor for neighbor in neighbors):
                peaks.append((current_height, (i, j)))
    # 按高度排序
    peaks.sort(reverse=True, key=lambda x: x[0])
    return peaks

# def get_peaks_bak(arr):
#     # 获取数组的维度
#     rows, cols = arr.shape
#     # 用一个列表来存储每个峰的高度及其坐标
#     peaks = []
#     for i in range(rows):
#         for j in range(cols):
#             # 获取当前元素的高度
#             current_height = arr[i, j]
#             neighbors = []
#             # 取出所有邻居，包括四个对角方向的邻居
#             if i > 0:
#                 neighbors.append(arr[i - 1, j])
#             if i < rows - 1:
#                 neighbors.append(arr[i + 1, j])
#             if j > 0:
#                 neighbors.append(arr[i, j - 1])
#             if j < cols - 1:
#                 neighbors.append(arr[i, j + 1])
#             if i > 0 and j > 0:
#                 neighbors.append(arr[i - 1, j - 1])
#             if i > 0 and j < cols - 1:
#                 neighbors.append(arr[i - 1, j + 1])
#             if i < rows - 1 and j > 0:
#                 neighbors.append(arr[i + 1, j - 1])
#             if i < rows - 1 and j < cols - 1:
#                 neighbors.append(arr[i + 1, j + 1])
#             # 判断当前元素是否大于周围的所有元素
#             if all(current_height > neighbor for neighbor in neighbors):
#                 peaks.append((current_height, (i, j)))
#     if len(peaks) < 3:
#         return None
#     # 按高度排序
#     peaks.sort(reverse=True, key=lambda x: x[0])
#     return peaks


def get_peak_3rd(peaks):
    peak_3rd = None
    for i in range(len(peaks)):
        peak = peaks[i]
        peak_val = peak[0]
        if peak_val < -3 and i > 1:
            peak_3rd = peak
            break
    return peak_3rd


def get_peak_5th(peaks):
    peak_5th = None
    for i in range(len(peaks)):
        peak = peaks[i]
        peak_val = peak[0]
        if peak_val < -3 and i > 4:
            peak_5th = peak
            break
    return peak_5th




if __name__ == '__main__':
    # 示例
    arr = np.array([[1.5, 2.3, 2.3, 3.1],
                    [4.2, 9.5, 5.5, 6.7],
                    [7.8, 8.8, 8.8, 9.9]])

    peaks = get_peaks(arr)
    print(peaks)