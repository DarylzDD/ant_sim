import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import chebwin
from mpl_toolkits.mplot3d import Axes3D

class ArrayChebyshev():

    # 计算切比雪夫振幅 - 面阵
    def getChebyshevPlane(self, Ny, Nz, SLL):
        Ty = chebwin(Ny, abs(SLL))
        Tz = chebwin(Nz, abs(SLL))
        # 生成权重矩阵
        weights = np.outer(Ty, Tz)
        #
        return weights

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

    # 打印切比雪夫幅度 - 面阵
    def printWeights(self, weights):
        print('阵元序列和对应的激励振幅:')
        for elemIdx, amplitude in enumerate(weights, start=1):
            print(f'第 #{elemIdx} 行的激励振幅: {amplitude}')

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


if __name__ == '__main__':
    print('test ArrayChebyshev:')
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
    SLL = -30
    #
    arrayChbyshev = ArrayChebyshev()
    #
    weights = arrayChbyshev.getChebyshevPlane(Ny, Nz, SLL)
    #
    arrayChbyshev.printWeights(weights)
    #
    pattern_dbw, theta, phi = arrayChbyshev.getPatternDbPlane(lambda_, Ny, Nz, d, weights, phi0, theta0, NA, NE, eps)
    #
    arrayChbyshev.drawPatternDb(pattern_dbw, theta, phi, theta0, phi0, NA, NE)