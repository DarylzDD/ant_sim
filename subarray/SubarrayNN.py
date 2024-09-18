import csv
import numpy as np
import matplotlib.pyplot as plt

from ArrayChebyshev import ArrayChebyshev
from util.util_msll_line import msll_line

def save_csv(data, file_path):
    # 指定CSV文件路径
    # file_path = 'data.csv'
    # 保存数据到CSV文件
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("数据已成功保存到CSV文件:", file_path)


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


class SubarrayNN():

    X = 3
    Y = 3

    ## 分割求均值
    def split_and_average(self, arr0, X, Y):
        arr1 = []
        for i in range(len(arr0)):
            arr1_x = []
            for j in range(len(arr0[0])):
                arr1_x.append(arr0[i][j])
            arr1.append(arr1_x)
        #
        m, n = len(arr1), len(arr1[0])  # 获取arr1的行数和列数
        sub_m, sub_n = m // X, n // Y  # 计算每个小二维列表的行数和列数
        # print("sub_m: %d, sub_n: %d" % (sub_m, sub_n))
        for i in range(0, m, sub_m):
            for j in range(0, n, sub_n):
                # print("i: %d, j: %d" % (i, j))
                list_avg = []
                for ii in range(i, i+sub_m):
                    for jj in range(j, j+sub_n):
                        # print(arr1[ii][jj])
                        list_avg.append(arr1[ii][jj])
                avg = np.mean(list_avg)
                for ii in range(i, i+sub_m):
                    for jj in range(j, j+sub_n):
                        arr1[ii][jj] = avg
                # print("list_avg:", list_avg, ", avg:", np.mean(list_avg))
        return arr1


    ## NN方式子阵划分
    def parse(self, Ny, Nz, SLL):
        # 1.设置基于切比雪夫的幅度
        arrayCheby = ArrayChebyshev()
        weights = arrayCheby.getChebyshevPlane(Ny, Nz, SLL)
        # 2.矩阵均分求平均
        # print("weights:")
        # print(weights)
        weights_avg = self.split_and_average(weights, self.X, self.Y)
        # print("weights_avg:")
        # print(weights_avg)
        # print("weights:")
        # print(weights)
        return np.array(weights_avg), weights

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
    print('test SubarrayNN:')
    #
    # 初始化参数
    lambda_ = 1  # 波长
    d = 0.5  # 阵元间隔
    Ny = 42  # 方位阵元个数
    Nz = 42  # 俯仰阵元个数
    phi0 = 0  # 方位指向
    theta0 = 0  # 俯仰指向
    eps = 0.0001  # 底电平
    NA = 360  # 方位角度采样
    NE = 360  # 俯仰角度采样
    SLL = -60
    #
    subarrayNN = SubarrayNN()
    # arr1 = [[1, 3, 5, 7, 8, 8], [1, 3, 5, 7, 8, 8], [8, 6, 1, 2, 7, 7], [8, 6, 4, 5, 7, 7]]
    # X = 2
    # Y = 3
    # print("arr1:", arr1)
    # arr2 = subarrayNN.split_and_average(arr1, X, Y)
    # print("arr2:", arr2)
    weights_avg, weights = subarrayNN.parse(Ny, Nz, SLL)
    print("weights_avg:")
    print(weights_avg)
    print("weights:")
    print(weights)
    save_csv(weights_avg, "subarrayNN_40x40-9_60_2024-06-28.csv")
    pattern_dbw, theta, phi = subarrayNN.getPatternDbPlane(lambda_, Ny, Nz, d, weights_avg, phi0, theta0, NA, NE, eps)
    #
    # subarrayNN.drawPatternDb(pattern_dbw, theta, phi, theta0, phi0, NA, NE)
    subarrayNN.drawWeight(weights_avg)
    #
    msll = psl_plane(NA, NE, theta0, phi0, pattern_dbw)
    print("msll:", msll)