import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def read_fBest_mat_item(path_file):
    # 读取.mat文件
    data = scipy.io.loadmat(path_file)
    # 将数据存储为列表
    my_list = data['fBest'].tolist()
    # print(my_list)
    list_data = []
    for li_data in my_list:
        for data in li_data:
            list_data.append(data)
    return list_data

def read_fBest_mat():
    # path_file_pre = "../files/subarray/dataset/chebyshev_thin_plane/fBest"
    # path_file_pre = "../files/subarray/dataset/chebyshev_thin_plane_40/fBest"
    # path_file_pre = "../files/subarray/dataset/chebyshev_thin_plane_40(30,30)/fBest"
    path_file_pre = "../files/subarray/dataset/chebyshev_thin_plane_40(30,0)/fBest"
    # path_file_pre = "../files/subarray/dataset/chebyshev_thin_plane_100/fBest"
    list_data = []
    # for i in range(100, 220, 5):
    for i in range(800, 1500, 10):
    # for i in range(5000, 7100, 100):
        path_file = path_file_pre+str(i)+".mat"
        print("path_file:", path_file)
        data = read_fBest_mat_item(path_file)
        list_data.append(data)
    return list_data

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


def convert_to_plane(arr_line, x, y):
    # 初始化一个填充了0的x行y列的二维数组
    arr_plane = [[0 for _ in range(y)] for _ in range(x)]
    # 遍历arr_line中的每个元素
    for val in arr_line:
        # 计算二维数组中的位置，将基于1的索引转换为基于0的索引
        # val - 1 是因为你的列表是从1开始计数的
        # 假设数组的布局是先行后列（即先填满一行，再移动到下一行）
        row_index = (val - 1) // y
        col_index = (val - 1) % y
        # 如果计算得到的索引位置在数组范围内，则设置为1
        if 0 <= row_index < x and 0 <= col_index < y:
            arr_plane[row_index][col_index] = 1
    return arr_plane


def find_ones_coordinates(arr_plane):
    # 初始化空的结果数组
    arr_plane2 = []
    # 遍历二维数组的每一行
    for i, row in enumerate(arr_plane):
        # 遍历行内的每个元素
        for j, val in enumerate(row):
            # 如果元素的值为1，记录其坐标
            if val == 1:
                arr_plane2.append([i, j])
    return arr_plane2

def count_arr():
    list_datas = read_fBest_mat()
    # for datas in list_datas:
    #     print(datas)
    result = [sum(x) for x in zip(*list_datas)]
    print("result:")
    # 打印宽度为3的列表下标
    print("[{}]".format(", ".join("{:<4}".format(i) for i, _ in enumerate(result))))
    # 打印宽度为3的列表值
    print("[{}]".format(", ".join("{:<4}".format(i) for i in result)))
    # print(result)
    sorted_indices = sorted(range(len(result)), key=lambda x: result[x], reverse=True)[:50]
    print("sorted_indices:")
    print(sorted_indices)
    #
    sorted_indices_plane = convert_to_plane(sorted_indices, 15, 15)
    print("sorted_indices_plane:")
    print(sorted_indices_plane)
    draw_map(sorted_indices_plane)
    #
    sorted_indices_plane_coor = find_ones_coordinates(sorted_indices_plane)
    print("sorted_indices_plane_coor:")
    print(sorted_indices_plane_coor)
    #
    # 第一次聚类
    kmeans1 = KMeans(n_clusters=15)
    labels1 = kmeans1.fit_predict(sorted_indices_plane_coor)
    centers1 = kmeans1.cluster_centers_
    print("聚类结果:")
    cluster_dict1 = {}
    for i, point in enumerate(sorted_indices_plane_coor):
        if labels1[i] not in cluster_dict1:
            cluster_dict1[labels1[i]] = {'points': [], 'center': None}
        cluster_dict1[labels1[i]]['points'].append((i, point))
        cluster_dict1[labels1[i]]['center'] = centers1[labels1[i]]
    coor_list = []
    for label, info in cluster_dict1.items():
        print("聚类标签:", label,"中心点:", info['center'])
        # print(' '.join(["数据点 {} : {}".format(index, point) for index, point in enumerate(info['points'])]))
        coor = int(int(info['center'][0]) * 15 + info['center'][1])
        print("coor:", coor)
        coor_list.append(coor)
    print("coor_list:")
    print(coor_list)
        # for index, point in info['points']:
        #     print("数据点", index, ":", point)
    #
    # plt.plot(result)
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.title('result')
    # plt.show()
    #
    x, y = 15, 15  # 设定行数和列数
    result2 = [result[i * y:(i + 1) * y] for i in range(x)]
    print("result2:")
    print(result2)
    draw_map(result2)


def get_spare_weight():
    # 1.读取稀疏结果
    list_datas = read_fBest_mat()
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
    # # debug: 画个图看看对不对
    # x, y = 15, 15  # 设定行数和列数
    # result2 = [result[i * y:(i + 1) * y] for i in range(x)]
    # print("result2:")
    # print(result2)
    # draw_map(result2)
    # 4.计算权重
    weight_list = []
    for r in result:
        weight_list.append(float(r/result_max))
    return weight_list


def test1():
    list1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x, y = 2, 5  # 设定行数和列数
    result = [list1[i * y:(i + 1) * y] for i in range(x)]
    print(result)


def test2():
    # 给定的二维数组
    list1 = [[0, 1, 2, 3, 2, 0], [1, 3, 1, 1, 2, 1]]
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

def test3():
    # 给定的一维数组和目标二维数组的大小
    arr_line = [4, 11, 1, 9, 10]
    x = 3
    y = 4
    # 转换并打印结果
    arr_plane = convert_to_plane(arr_line, x, y)
    for row in arr_plane:
        print(row)

def test4():
    # 给定的二维数组
    arr_plane = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 1]]
    # 转换并打印结果
    arr_plane2 = find_ones_coordinates(arr_plane)
    print(arr_plane2)


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    # count_arr()
    # list_data = read_fBest_mat()
    # print("list_data:")
    # print(list_data)
    weighted_spare_list = get_spare_weight()
    result_sorted = sorted(weighted_spare_list, reverse=True)
    result_max = result_sorted[0]
    print("weighted_spare_list_max:", result_max)
    print("weighted_spare_list_min:", result_sorted[-1])