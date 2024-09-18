import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import chebwin


def arr_chebyshev(lambda_, N, SLL, d):
    # 计算波数
    k = 2 * np.pi / lambda_
    # 切比雪夫窗
    a = chebwin(N, abs(SLL))
    # 输出每个阵元的序列号和激励振幅
    # print('阵元序列和对应的激励振幅:')
    # for elemIdx, amplitude in enumerate(a, start=1):
    #     print(f'阵元 #{elemIdx} 的激励振幅: {amplitude}')
    # 角度范围从-180度到180度（以弧度为单位）
    # theta = np.linspace(-np.pi, np.pi, 1000)
    # 角度范围从-90度到90度（以弧度为单位）
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    # 初始化阵因子Array Factor(AF)
    AF = np.zeros(theta.shape, dtype=complex)
    # 计算阵因子
    for i, ang in enumerate(theta):
        elementPos = np.arange(N)  # 阵元位置矢量
        phaseDiff = k * d * np.cos(np.pi / 2 - ang)
        # phaseDiff = k * d * np.cos(ang)
        AF[i] = np.sum(a * np.exp(1j * phaseDiff * elementPos))
    # 归一化阵因子
    AF = np.abs(AF) / np.max(np.abs(AF))
    # 转换角度到度
    theta_deg = np.rad2deg(theta)
    #
    return a, AF, theta_deg


def print_w(a):
    print('阵元序列和对应的激励振幅:')
    for elemIdx, amplitude in enumerate(a, start=1):
        print(f'阵元 #{elemIdx} 的激励振幅: {amplitude}')


def draw_af_theta(AF, theta_deg):
    # 绘制方向图
    plt.figure()
    plt.plot(theta_deg, 20 * np.log10(AF))
    plt.xlabel('theta-degree')
    plt.ylabel('gain-dB')
    plt.title('chebyshev')
    plt.grid(True)
    plt.xlim([-90, 90])
    plt.ylim([20 * np.log10(np.max(AF)) - 80, 0])
    # 显示图表
    plt.show()


if __name__ == '__main__':
    print('arr chebyshev')
    #
    # 阵列参数
    N = 12  # 阵元数
    d = 0.5  # 阵元间距（以波长为单位）
    lambda_ = 1  # 波长
    SLL = -35  # 副瓣电平，以分贝为单位
    #
    a, AF, theta_deg = arr_chebyshev(lambda_, N, SLL, d)
    #
    print_w(a)
    #
    draw_af_theta(AF, theta_deg)