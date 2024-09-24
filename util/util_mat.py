import scipy.io


# 读取.mat文件
def read_mat(path_file):
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