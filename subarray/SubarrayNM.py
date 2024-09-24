import csv
import numpy as np
import matplotlib.pyplot as plt

from ArrayChebyshev import ArrayChebyshev
from util.util_analysis_line import msll_line




def psl_plane(NA, NE, theta0, phi0, pattern_dbw):
    temp1 = pattern_dbw[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
    temp2 = pattern_dbw[round(NA * ((np.pi / 2 + phi0) / np.pi)), :]
    # print("temp1:", temp1)
    # print("temp2:", temp2)
    msll_arr_w_1 = msll_line(temp1)
    msll_arr_w_2 = msll_line(temp2)
    msll_arr_w = max(msll_arr_w_1, msll_arr_w_2)
    #
    return msll_arr_w

def save_csv(data, file_path):
    # 指定CSV文件路径
    # file_path = 'data.csv'
    # 保存数据到CSV文件
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("数据已成功保存到CSV文件:", file_path)




class SubarrayNM():
    K = 9

    def __assign_nearest_index(self, weights, interval_array):
        nearest_indices = np.zeros_like(weights)
        # 遍历所有元素
        for i in range(weights.shape[0]):  # 遍历行
            for j in range(weights.shape[1]):  # 遍历列
                nearest = 0
                for k in range(interval_array.shape[0]):
                    if np.abs(weights[i, j]-interval_array[k]) < np.abs(weights[i, j]-interval_array[nearest]):
                        nearest = k
                nearest_indices[i, j] = nearest
        return nearest_indices


    def __assign_nearest_index_bak(self, weights, interval_array):
        """
        将weights中的每个元素替换为其在interval_array中最接近值的索引。
        参数:
        weights -- 类型为np.array的二维数组，尺寸为MxN
        interval_array -- 一维等间隔数组，由np.linspace生成
        返回:
        更新后的weights，其中每个元素被替换为它在interval_array中最接近值的索引。
        """
        # 确保interval_array的长度比weights中的元素多，这样才能为每个元素找到一个最近的值
        assert interval_array.shape[0] >= weights.size, "interval_array的长度需要至少与weights中的元素数量一样多"
        # 将weights展平以便更容易地进行比较
        flat_weights = weights.flatten()
        # 初始化一个数组用于存储索引
        nearest_indices = np.zeros_like(flat_weights, dtype=int)
        # 对weights中的每个值，找到其在interval_array中最近的值的索引
        for i, weight in enumerate(flat_weights):
            # 计算与每个区间值的差的绝对值，找到最小差值的索引
            nearest_index = np.argmin(np.abs(interval_array - weight))
            nearest_indices[i] = nearest_index
        # 将一维的nearest_indices重塑回与原weights相同的形状
        return nearest_indices.reshape(weights.shape)


    def __average_values_by_indices(self, weights, weights_indices):
        """
        根据weights_indices中的索引，对weights中的值进行分组平均，
        并将结果存储到一个新的与weights同尺寸的数组weights_sub中。
        参数:
        weights -- 类型为np.array的二维数组，尺寸为MxN
        weights_indices -- 与weights同尺寸的二维数组，存储了每个元素应归属的组的索引
        返回:
        weights_sub -- 与weights同尺寸的二维数组，其中每个值是对应位置weights中相同索引值的平均值
        """
        # 获取weights的尺寸
        M, N = weights.shape
        # 初始化weights_sub为weights的形状，初始填充NaN以方便后续检查是否所有索引都被处理
        weights_sub = np.full_like(weights, np.nan)
        # 遍历所有可能的索引值
        unique_indices = np.unique(weights_indices)
        for index in unique_indices:
            # 找到weights_indices中等于当前index的所有位置
            mask = weights_indices == index
            # 根据位置mask选择weights中的值，然后计算这些值的平均值
            avg_value = np.mean(weights[mask])
            # 将平均值赋给weights_sub中对应的位置
            weights_sub[mask] = avg_value
        # 检查weights_sub中是否还有NaN值，确保所有位置都被正确处理
        if np.isnan(weights_sub).any():
            raise ValueError("有些索引可能没有对应的值进行平均，请检查weights_indices是否包含了所有必要的索引。")
        return weights_sub


    ## NM方式子阵划分 (Nickel's Method)
    def parse(self, Ny, Nz, SLL):
        # 1.设置基于切比雪夫的幅度
        arrayCheby = ArrayChebyshev()
        weights = arrayCheby.getChebyshevPlane(Ny, Nz, SLL)
        # 2.求所有阵元激励的最大值和最小值, 并平均分间隔
        weights_max = np.max(weights)
        weights_min = np.min(weights)
        # 使用linspace生成从最小值到最大值的K个等间距点，注意 endpoints 参数设为 False，因为两端值我们分别想要的是 min_value 和 max_value
        interval_array = np.linspace(weights_min, weights_max, self.K, endpoint=False)
        # 由于linspace默认包含端点，为了确保两端值准确，我们手动添加第一个值为min_value，最后一个值为max_value
        if interval_array.size < self.K:  # 确保K的大小合理，至少为2
            interval_array = np.concatenate(([weights_min], interval_array, [weights_max]))
        else:  # 如果K足够大，直接替换两端值
            interval_array[0] = weights_min
            interval_array[-1] = weights_max
        print("生成的等间距数组:", interval_array)
        # 3.把weights根据距离interval_array中值的远近分到不同的组
        weights_indices = self.__assign_nearest_index(weights, interval_array)
        # 4.weights_indices同位置取平均值给weights_sub
        weights_sub = self.__average_values_by_indices(weights, weights_indices)
        print("原始weights:")
        print(weights)
        np.set_printoptions(threshold=np.inf)
        print("\nweights_indices:")
        print(weights_indices)
        print("\n根据weights_indices求得的weights_sub:")
        print(weights_sub)
        #
        return weights_sub, weights


    # 打印切比雪夫幅度 - 面阵
    def printWeights(self, weights):
        print('阵元序列和对应的激励振幅:')
        for elemIdx, amplitude in enumerate(weights, start=1):
            print(f'第 #{elemIdx} 行的激励振幅: {amplitude}')

    # 根据切比雪夫振幅, 计算方向图Pattern - 面阵
    def getPatternDbPlane(self, lambda_, Ny, Nz, d, weights, phi0, theta0, NA, NE, eps):
        phi = np.linspace(-np.pi / 2, np.pi / 2, NA)
        theta = np.linspace(-np.pi / 2, np.pi / 2, NE)
        aa = np.arange(0, d * Ny, d)
        bb = np.arange(0, d * Nz, d)
        DD1 = np.repeat(aa[:, np.newaxis], Nz, axis=1)
        DD2 = np.repeat(bb[np.newaxis, :], Ny, axis=0)
        DD = DD1 + 1j * DD2
        #
        pattern = np.zeros((len(phi), len(theta)), dtype=complex)
        #
        for jj in range(len(phi)):
            for ii in range(len(theta)):
                pattern0 = weights * np.exp(1j * 2 * np.pi / lambda_ *
                                            (np.sin(phi[jj]) * np.cos(theta[ii]) * DD1 +
                                             np.sin(theta[ii]) * DD2 -
                                             np.sin(phi0) * np.cos(theta0) * DD1 -
                                             np.sin(theta0) * DD2))
                pattern[jj, ii] = np.sum(pattern0)
        #
        max_p = np.max(np.abs(pattern))
        pattern_dbw = 20 * np.log10(np.abs(pattern) / max_p + eps)
        #
        return pattern_dbw, theta, phi

    # 画图 - 方向图
    def drawPatternDb(self, pattern_dbw, theta, phi, theta0, phi0, NA, NE):
        pattern_dbw[pattern_dbw < -50] = -50 + np.random.uniform(-1, 1, np.sum(pattern_dbw < -50))
        # 绘制方向图
        phi_deg = phi * 180 / np.pi
        theta_deg = theta * 180 / np.pi
        Phi, Theta = np.meshgrid(phi_deg, theta_deg)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Phi, Theta, pattern_dbw.T, cmap='viridis')  # 注意 pattern_dbw 需转置
        ax.set_xlabel('Phi (Degrees)')
        ax.set_ylabel('Theta (Degrees)')
        ax.set_title('Dolph-Chebyshev Planar Array Radiation Pattern')
        # 绘制方向图
        # plt.figure()
        # plt.pcolormesh(theta * 180 / np.pi, phi * 180 / np.pi, pattern_dbw, shading='auto')
        # plt.colorbar()
        # plt.xlabel('Phi (Degrees)')
        # plt.ylabel('Theta (Degrees)')
        # plt.title('Dolph-Chebyshev Planar Array Radiation Pattern')
        # plt.tight_layout()
        # 绘制方位向切面图
        plt.figure()
        temp1 = pattern_dbw[:, round(NE * ((np.pi / 2 + theta0) / np.pi))]
        plt.plot(phi * 180 / np.pi, temp1)
        plt.grid()
        plt.xlabel('Phi (degree)')
        plt.ylabel('gain (dB)')
        # 绘制俯仰向切面图
        plt.figure()
        temp2 = pattern_dbw[round(NA * ((np.pi / 2 + phi0) / np.pi)), :]
        plt.plot(theta * 180 / np.pi, temp2)
        plt.grid()
        plt.xlabel('Phi (degree)')
        plt.ylabel('gain (dB)')
        plt.show()
        # 绘制方向图
        # phi_mesh, theta_mesh = np.meshgrid(phi, theta)
        # X = np.sin(phi_mesh) * np.cos(theta_mesh)
        # Y = np.sin(phi_mesh) * np.sin(theta_mesh)
        # Z = np.cos(phi_mesh)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(X, Y, pattern_dbw.T, cmap='viridis', edgecolor='none')  # 注意pattern_dbw需转置
        # fig.colorbar(surf)
        # ax.set_xlabel('theta')
        # ax.set_ylabel('phi')
        # ax.set_zlabel('gain(dB)')
        # ax.set_title('Dolph-Chebyshev Planar Array Radiation Pattern')
        # plt.show()

    def drawWeight(self, data):
        plt.figure()
        plt.imshow(data)
        plt.axis('off')  # 这将隐藏x轴和y轴
        plt.show()




if __name__ == '__main__':
    print("subarrayNM")
    #
    # 初始化参数
    lambda_ = 1  # 波长
    d = 0.5  # 阵元间隔
    Ny = 40  # 方位阵元个数
    Nz = 40  # 俯仰阵元个数
    phi0 = 0  # 方位指向
    theta0 = 0  # 俯仰指向
    eps = 0.0001  # 底电平
    NA = 360  # 方位角度采样
    NE = 360  # 俯仰角度采样
    SLL = -60
    #
    subarrayNM = SubarrayNM()
    weights_sub, weights = subarrayNM.parse(Ny, Nz, SLL)
    print("weights_sub:")
    print(weights_sub)
    print("weights:")
    print(weights)
    #
    pattern_dbw, theta, phi = subarrayNM.getPatternDbPlane(lambda_, Ny, Nz, d, weights_sub, phi0, theta0, NA, NE, eps)
    #
    subarrayNM.drawWeight(weights_sub)
    # save_csv(weights_sub, "subarrayNM_40x40-9_60_2024-06-27.csv")
    save_csv(weights_sub, "../files/subarrayNM_40x40-9_60_2024-06-27.csv")
    #
    msll = psl_plane(NA, NE, theta0, phi0, pattern_dbw)
    print("msll:", msll)