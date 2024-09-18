import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 参数设置
lambda_ = 1  # 波长
d = lambda_ / 2  # 单元间距
N = 10  # 水平方向上的单元数
M = 10  # 垂直方向上的单元数

# 波束指向角 (theta0, phi0)
theta0 = np.pi / 24  # 7.5度
phi0 = np.pi / 4  # 45度
# theta0 = np.pi / 12  # 15度
# phi0 = np.pi / 4  # 45度

# 计算相位偏移
kx = 2 * np.pi / lambda_ * np.sin(theta0) * np.cos(phi0)
ky = 2 * np.pi / lambda_ * np.sin(theta0) * np.sin(phi0)

# 计算theta和phi的范围
theta = np.linspace(0, np.pi/2, 500)
phi = np.linspace(-np.pi, np.pi, 500)
Theta, Phi = np.meshgrid(theta, phi)

# 将theta和phi转换为u和v
U = np.sin(Theta) * np.cos(Phi)
V = np.sin(Theta) * np.sin(Phi)

# 初始化方向图
AF = np.zeros(U.shape, dtype=complex)

# 计算阵列因子
for n in range(N):
    for m in range(M):
        AF += np.exp(1j * 2 * np.pi / lambda_ * (n * d * U + m * d * V) - 1j * (n * kx + m * ky))

# 取绝对值并归一化
AF = np.abs(AF)
AF = AF / np.max(AF)

# 绘制方向图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Theta * 180/np.pi, Phi * 180/np.pi, AF, cmap='viridis')

ax.set_xlabel('Theta (degrees)')
ax.set_ylabel('Phi (degrees)')
ax.set_zlabel('Normalized AF')
ax.set_title('Uniform Rectangular Planar Array Pattern with Beam Steering')

plt.show()


max_index = np.unravel_index(np.argmax(AF), AF.shape)
print("max_index: %f, %f" % (max_index[0], max_index[1]))
temp2_1 = AF[max_index[0], :]
temp2_2 = AF[:, max_index[1]]
plt.figure()
plt.plot(theta * 180 / np.pi, temp2_1, label='temp2_1', color='#ff7f0e')
plt.grid()
plt.legend()
plt.xlabel('theta (degree)')
plt.ylabel('normalized pattern (dB)')
plt.show()
#
plt.figure()
plt.plot(phi * 180 / np.pi, temp2_2, label='temp2_2', color='#ff7f0e')
plt.grid()
plt.legend()
plt.xlabel('phi (degree)')
plt.ylabel('normalized pattern (dB)')
plt.show()

