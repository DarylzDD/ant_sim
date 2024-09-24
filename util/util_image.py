import matplotlib.pyplot as plt

# ============================================= 图片处理相关 ========================================================
# 保存图片
def draw_img(data):
    plt.figure()
    plt.imshow(data)
    plt.axis('off')  # 这将隐藏x轴和y轴
    # plt.colorbar()
    # plt.title('mag_E')
    # plt.xlabel('Theta')
    # plt.ylabel('Phi')
    plt.show()

def save_img(path_img, data):
    plt.figure()
    plt.imshow(data)
    plt.axis('off')  # 这将隐藏x轴和y轴
    # plt.colorbar()
    # plt.title('mag_E')
    # plt.xlabel('Theta')
    # plt.ylabel('Phi')
    # plt.show()
    # plt.savefig(path_img, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(path_img, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig=None)  # 关闭当前图像窗口，如果fig=None，则默认关闭最近打开的图像窗口


def draw_img_xyz(data, x, y):
    # 绘制图像
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x, y, data, shading='auto', cmap='viridis')
    # plt.pcolormesh(x, y, np.abs(pattern_xyz), shading='auto', cmap='viridis')
    # plt.colorbar(label='Pattern Magnitude')
    plt.axis('equal')
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()
    # plt.savefig(path_img, dpi=200, bbox_inches='tight', pad_inches=0)
    # plt.close(fig=None)  # 关闭当前图像窗口，如果fig=None，则默认关闭最近打开的图像窗口


def save_img_xyz(path_img, data, x, y):
    # 绘制图像
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x, y, data, shading='auto', cmap='viridis')
    # plt.pcolormesh(x, y, np.abs(pattern_xyz), shading='auto', cmap='viridis')
    # plt.colorbar(label='Pattern Magnitude')
    plt.axis('equal')
    plt.axis('off')  # 关闭坐标轴显示
    # plt.show()
    plt.savefig(path_img, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig=None)  # 关闭当前图像窗口，如果fig=None，则默认关闭最近打开的图像窗口
