import h5py

# =================================================== 以下是工具类方法 ==============================================
# 读取 .h5py
def read_h5(path_h5):
    # 指定HDF5文件的路径
    # path_h5 = 'dataset.h5'
    # 使用h5py的File函数以读取模式打开HDF5文件
    with h5py.File(path_h5, 'r') as hf:
        # 从文件中读取inputs和outputs数据集
        inputs = hf['inputs'][()]
        outputs = hf['outputs'][()]
        # 确认数据已正确读取
        print("Inputs shape:", inputs.shape)
        print("Outputs shape:", outputs.shape)
        # 此处可以进行进一步的数据处理或分析
        # 例如：打印部分数据检查
        # print("First element of Inputs:", inputs[0])
        # print("First element of Outputs:", outputs[0])
        # 返回
        return inputs, outputs


# 保存 .h5py
def save_h5(path_h5, X, Y):
    # 假设 X 是一个形状为 (60000, 500, 500) 的numpy数组，Y 是形状为 (60000, 8, 8) 的numpy数组
    # with h5py.File('dataset.h5', 'w') as hf:
    with h5py.File(path_h5, 'w') as hf:
        hf.create_dataset('inputs', data=X, compression="gzip", compression_opts=9)
        hf.create_dataset('outputs', data=Y, compression="gzip", compression_opts=9)