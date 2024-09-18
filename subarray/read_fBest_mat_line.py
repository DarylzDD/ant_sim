import scipy.io
import matplotlib.pyplot as plt

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
    # path_file_pre = "../files/subarray/dataset/chebyshev_thin_line/fBest"
    # path_file_pre = "../files/subarray/dataset/chebyshev_thin_line_100(0)/fBest"
    # path_file_pre = "../files/subarray/dataset/chebyshev_thin_line_100(30)/fBest"
    path_file_pre = "../files/subarray/dataset/chebyshev_thin_line_100(15)/fBest"
    list_data = []
    for i in range(20, 100, 5):
        path_file = path_file_pre+str(i)+".mat"
        # print("path_file:", path_file)
        data = read_fBest_mat_item(path_file)
        list_data.append(data)
    return list_data

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

def count_arr():
    list_datas = read_fBest_mat()
    for datas in list_datas:
        print(datas)
    result = [sum(x) for x in zip(*list_datas)]
    print("result:")
    # 打印宽度为3的列表下标
    print("[{}]".format(", ".join("{:<3}".format(i) for i, _ in enumerate(result))))
    # 打印宽度为3的列表值
    print("[{}]".format(", ".join("{:<3}".format(i) for i in result)))
    # print(result)
    sorted_indices = sorted(range(len(result)), key=lambda x: result[x], reverse=True)[:9]
    print(sorted_indices)
    plt.plot(result)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('result')
    plt.show()


def test():
    li = [[0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [1, 1, 0, 1, 0],
          [1, 1, 0, 0, 0],
          [0, 1, 0, 1, 0],]
    result = [sum(x) for x in zip(*li)]
    print(result)


if __name__ == '__main__':
    # test()
    count_arr()