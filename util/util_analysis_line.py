import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import chebwin


def msll_line(a):
    extremes = [0]
    # a = AF_dB
    for i in range(1, len(a) - 1):
        if (a[i] > a[i - 1] and a[i] > a[i + 1]) or (a[i] < a[i - 1] and a[i] < a[i + 1]):
            extremes.append(i)
    extremes.append(len(a) - 1)
    # print("extremes:", extremes)
    peek_val = []
    for i in extremes:
        if a[i] < -3:
            peek_val.append(a[i])
            # print(a[i])
    sorted_peek_val = sorted(peek_val)
    # print(sorted_peek_val)
    second_largest_value = sorted_peek_val[-1]
    second_largest_index = peek_val.index(second_largest_value)
    # print("第二大的值:", second_largest_value)
    # print("对应的序号:", second_largest_index)
    return second_largest_value




if __name__ == '__main__':
    # 阵列参数
    N = 8  # 阵元数
    d = 0.5  # 阵元间距（以波长为单位）
    lambda_ = 1  # 波长
    SLL = -30  # 副瓣电平，以分贝为单位

    # 计算波数
    k = 2 * np.pi / lambda_

    # 切比雪夫窗
    a = chebwin(N, abs(SLL))

    # 输出每个阵元的序列号和激励振幅
    print('阵元序列和对应的激励振幅:')
    for elemIdx, amplitude in enumerate(a, start=1):
        print(f'阵元 #{elemIdx} 的激励振幅: {amplitude}')

    # 计算最大旁瓣电平
    MSLL = np.max(a) - np.mean(a)
    print('最大旁瓣电平:', MSLL)


    # 角度范围从-180度到180度（以弧度为单位）
    theta = np.linspace(-np.pi, np.pi, 1000)
    # 初始化阵因子Array Factor(AF)
    AF = np.zeros(theta.shape, dtype=complex)

    # 计算阵因子
    for i, ang in enumerate(theta):
        elementPos = np.arange(N)  # 阵元位置矢量
        phaseDiff = k * d * np.cos(np.pi/2 - ang)
        AF[i] = np.sum(a * np.exp(1j * phaseDiff * elementPos))

    # 归一化阵因子
    AF = np.abs(AF) / np.max(np.abs(AF))

    # 转换角度到度
    theta_deg = np.rad2deg(theta)

    # 归一化阵因子
    # AF_abs = np.abs(AF) / np.max(np.abs(AF))
    AF_abs = AF
    AF_dB = 20 * np.log10(AF_abs)  # 转换为dB


    # 计算最大旁瓣电平
    # MSLL_AF_dB = np.max(AF_dB) - np.mean(AF_dB)
    # print('最大旁瓣电平MSLL_AF_dB:', MSLL_AF_dB)

    msll = msll_line(AF_dB)
    print(f'msll（最大旁瓣电平）：{msll:.6f} dB')


    # 绘制方向图
    plt.figure()
    plt.plot(theta_deg, 20*np.log10(AF))
    plt.xlabel('theta-degree')
    plt.ylabel('gain-dB')
    plt.title('chebyshev')
    plt.grid(True)
    plt.xlim([-180, 180])
    plt.ylim([20*np.log10(np.max(AF)) - 50, 0])

    # 显示图表
    plt.show()

    print("fin.")