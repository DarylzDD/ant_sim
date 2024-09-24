import numpy as np

from math import sin, cos, radians, sqrt


# 初始化参数
nRow = 64  # x方向单元个数，θ方向
mLine = 64  # y方向单元个数，φ方向
f0 = 300  # 工作频点，单位：GHz
c = 3e8  # 光速，单位：m/s
lamda0 = c / (f0 * 1e9)  # 波长，单位：m
dx = lamda0 / 2  # x方向单元间距，单位：m
dy = dx  # y方向单元间距，单位：m
k = 2 * np.pi / lamda0  # 波数

# 配置馈元和计算馈元位置
InF = 1  # 喇叭馈源摆放状态，0：垂直阵面，1：指向阵面中心
InTetha = 30  # 入射角
aphea_10dB = 45  # 喇叭馈源的-10dB波束角宽度

# 计算辐射方向图
qf = 1
qe = 1
eps = 0.000001



# ============================================= 方向图计算 -- 基础方法 ====================================
# 计算距离
def distance(p1, p2):
    return sqrt(sum((np.array(p1) - np.array(p2)) ** 2))

# 计算相位
def get_phase(posx, posy, beamvec, Feed, Rc, k):
    Nx, Ny = len(posx), len(posy)
    phase = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            E = np.array([posx[i], posy[j], 0])
            rfij = distance(Feed, E)
            rf = distance(Feed, Rc)
            phase[i, j] = k * (rfij - rf - np.dot(beamvec, E))
    phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
    return phase

# 计算馈元
def get_feed(InF, InTetha, aphea_10dB, nRow, dx):
    D = nRow * dx
    if InF == 0:
        beata = 90 - aphea_10dB / 2
        gema = 90 - beata - InTetha
        F = sin(radians(beata)) * ((D / 2) / sin(radians(gema)))
        xf = F * sin(radians(InTetha))
        yf = 0
        zf = F * cos(radians(InTetha))
    elif InF == 1:
        beata = 90 - InTetha
        gema = 180 - beata - aphea_10dB / 2
        F = sin(radians(gema)) * ((D / 2) / sin(radians(aphea_10dB / 2)))
        xf = F * sin(radians(InTetha))
        yf = 0
        zf = F * cos(radians(InTetha))
    return [xf, yf, zf], F / D

# 计算方向图
def get_pattern(phase, posx, posy, k, Feed, Rc, qf, qe):
    """
    GET_PATTERN 得到3维方向图
    """
    Nx, Ny = len(posx), len(posy)
    div = 361
    theta = np.linspace(0, np.pi / 2, div)
    phi = np.linspace(0, 2 * np.pi * (1 - 1 / div), div - 1)
    th, ph = np.meshgrid(theta, phi)

    pattern = np.zeros_like(th, dtype=complex)
    for i in range(Nx):
        for j in range(Ny):
            E = np.array([posx[i], posy[j], 0])
            rfij = distance(Feed, E)

            # 简化处理，假设单元方向图A为1
            A = 1

            p = (-k * rfij + phase[i, j] +
                 k * (posx[i] * np.sin(th) * np.cos(ph) +
                      posy[j] * np.sin(th) * np.sin(ph)))
            p = A * np.exp(1j * p)
            pattern += p

    # 这一步可以根据实际需求选择是否执行
    # pattern = np.cos(th) ** qe * pattern
    return pattern, th, ph


# ============================================= 方向图计算 -- 应用方法 ====================================
# 计算方向图
def parse_pattern_dbw(phaseBit):
    # # 初始化参数
    # nRow = 64  # x方向单元个数，θ方向
    # mLine = 64  # y方向单元个数，φ方向
    # f0 = 300  # 工作频点，单位：GHz
    # c = 3e8  # 光速，单位：m/s
    # lamda0 = c / (f0 * 1e9)  # 波长，单位：m
    # dx = lamda0 / 2  # x方向单元间距，单位：m
    # dy = dx  # y方向单元间距，单位：m
    # k = 2 * np.pi / lamda0  # 波数
    # 生成天线位置坐标
    posx = np.arange(-dx * (nRow - 1) / 2, dx * (nRow - 1) / 2 + dx, dx)
    posy = np.arange(-dy * (mLine - 1) / 2, dy * (mLine - 1) / 2 + dy, dy)
    # # 配置馈元和计算馈元位置
    # InF = 1  # 喇叭馈源摆放状态，0：垂直阵面，1：指向阵面中心
    # InTetha = 30  # 入射角
    # aphea_10dB = 45  # 喇叭馈源的-10dB波束角宽度
    Feed, FD = get_feed(InF, InTetha, aphea_10dB, nRow, dx)
    # 参考阵元位置
    Rc = [posx[0], posy[0], 0]
    # 计算辐射方向图
    # qf = 1
    # qe = 1
    # eps = 0.000001
    phaseDeg = phaseBit * 180
    phaseRad = np.deg2rad(phaseDeg)
    pattern, th, ph = get_pattern(phaseRad, posx, posy, k, Feed, Rc, qf, qe)
    # 方向图归一化
    max_p = np.max(np.max(np.abs(pattern)))
    pattern_dbw = 20 * np.log10(np.abs(pattern) / max_p + eps)
    #
    return pattern_dbw


# 相位生成方向图
def phase_2_pattern(phaseBit):
    # # 初始化参数
    # nRow = 64  # x方向单元个数，θ方向
    # mLine = 64  # y方向单元个数，φ方向
    # f0 = 300  # 工作频点，单位：GHz
    # c = 3e8  # 光速，单位：m/s
    # lamda0 = c / (f0 * 1e9)  # 波长，单位：m
    # dx = lamda0 / 2  # x方向单元间距，单位：m
    # dy = dx  # y方向单元间距，单位：m
    # k = 2 * np.pi / lamda0  # 波数
    # qf = 1
    # qe = 1

    # 生成天线位置坐标
    posx = np.arange(-dx * (nRow - 1) / 2, dx * (nRow - 1) / 2 + dx, dx)
    posy = np.arange(-dy * (mLine - 1) / 2, dy * (mLine - 1) / 2 + dy, dy)

    # 参考阵元位置
    Rc = [posx[0], posy[0], 0]

    # # 配置馈元和计算馈元位置
    # InF = 1  # 喇叭馈源摆放状态，0：垂直阵面，1：指向阵面中心
    # InTetha = 30  # 入射角
    # aphea_10dB = 45  # 喇叭馈源的-10dB波束角宽度
    Feed, FD = get_feed(InF, InTetha, aphea_10dB, nRow, dx)

    # 计算方向图
    pattern, th, ph = get_pattern(phaseBit, posx, posy, k, Feed, Rc, qf, qe)
    pattern = np.abs(pattern)

    return pattern

# 相位生成方向图, xy坐标系
def phase_2_pattern_xyz(phaseBit):
    # 1. 相位生成方向图, 极坐标
    # 生成天线位置坐标
    posx = np.arange(-dx * (nRow - 1) / 2, dx * (nRow - 1) / 2 + dx, dx)
    posy = np.arange(-dy * (mLine - 1) / 2, dy * (mLine - 1) / 2 + dy, dy)
    # 参考阵元位置
    Rc = [posx[0], posy[0], 0]
    # 计算馈元位置
    Feed, FD = get_feed(InF, InTetha, aphea_10dB, nRow, dx)
    # 计算方向图
    pattern, th, ph = get_pattern(phaseBit, posx, posy, k, Feed, Rc, qf, qe)
    # 2. 转换为直角坐标系
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = np.cos(th)
    return pattern, x, y, z


# 角度规范化到 (-180, 180) 范围
def normalize_angle(angle):
    # 将角度规范化到 [0, 360) 范围
    angle = angle % 360
    # 如果规范化后的角度大于 180，则转换到 (-180, 180) 范围
    angle = np.where(angle > 180, angle - 360, angle)
    return angle


# 相位转换 X bit
def phase_2_bit(phase, bit_num):
    phase = normalize_angle(phase)
    if bit_num == 1:
        # 相位转换1bit -- 1bit
        phaseBit = np.where(np.logical_or(phase >= 90, phase <= -90), 1, 0)
        # 比特转角度
        phaseDeg = phaseBit * 180
    elif bit_num == 2:
        # 相位转换2bit
        phaseBit = np.where((phase >= -180) & (phase < -90), -3,
                            np.where((phase >= -90) & (phase < 0), -1,
                                     np.where((phase >= 0) & (phase < 90), 1,
                                              np.where((phase >= 90) & (phase <= 180), 3, 0))))
        # 将0的情况去掉，理论上不应该出现这种情况
        if np.any(phaseBit == 0):
            # 使用 np.where 找到值为 0 的点的坐标
            zero_points = np.where(phase == 0)
            # zero_points 是一个元组，包含了行和列的索引
            # 将其转换为坐标列表
            coordinates = list(zip(zero_points[0], zero_points[1]))
            # 打印结果
            print("Coordinates of elements equal to 0:", coordinates)
            raise ValueError("There are phase values outside the expected range of -180 to 180.")
        # 比特转角度
        phaseDeg = phaseBit * 45
    else:
        # 默认相位转换1bit -- 1bit
        phaseBit = np.where(np.logical_or(phase >= 90, phase <= -90), 1, 0)
        # 比特转角度
        phaseDeg = phaseBit * 180
    return phaseBit, phaseDeg


# 主方法: 波束指向角生成相位和方向图
def point_2_phi_pattern(theta0, phi0, bit_num=1):
    # # 初始化参数
    # nRow = 64  # x方向单元个数，θ方向
    # mLine = 64  # y方向单元个数，φ方向
    # f0 = 300  # 工作频点，单位：GHz
    # c = 3e8  # 光速，单位：m/s
    # lamda0 = c / (f0 * 1e9)  # 波长，单位：m
    # dx = lamda0 / 2  # x方向单元间距，单位：m
    # dy = dx  # y方向单元间距，单位：m
    # k = 2 * np.pi / lamda0  # 波数
    beamvec = [sin(radians(theta0)) * cos(radians(phi0)),
               sin(radians(theta0)) * sin(radians(phi0)), 0]

    # 生成天线位置坐标
    posx = np.arange(-dx * (nRow - 1) / 2, dx * (nRow - 1) / 2 + dx, dx)
    posy = np.arange(-dy * (mLine - 1) / 2, dy * (mLine - 1) / 2 + dy, dy)

    # 参考阵元位置
    Rc = [posx[0], posy[0], 0]

    # # 配置馈元和计算馈元位置
    # InF = 1  # 喇叭馈源摆放状态，0：垂直阵面，1：指向阵面中心
    # InTetha = 30  # 入射角
    # aphea_10dB = 45  # 喇叭馈源的-10dB波束角宽度
    Feed, FD = get_feed(InF, InTetha, aphea_10dB, nRow, dx)

    # 计算反射面相位分布
    phase = get_phase(posx, posy, beamvec, Feed, Rc, k)

    # 将相位从弧度转角度
    phase = np.rad2deg(phase)

    # 连续相位转 X bit
    phaseBit, phaseDeg = phase_2_bit(phase, bit_num)
    phaseRad = np.deg2rad(phaseDeg)

    # 计算方向图
    pattern, th, ph = get_pattern(phaseRad, posx, posy, k, Feed, Rc, qf, qe)
    pattern = np.abs(pattern)

    return phase, phaseBit, pattern


# 主方法: 波束指向角生成相位 (注意返回弧度还是角度)
def point_2_phase(theta0, phi0):
    # # 初始化参数
    # nRow = 64  # x方向单元个数，θ方向
    # mLine = 64  # y方向单元个数，φ方向
    # f0 = 300  # 工作频点，单位：GHz
    # c = 3e8  # 光速，单位：m/s
    # lamda0 = c / (f0 * 1e9)  # 波长，单位：m
    # dx = lamda0 / 2  # x方向单元间距，单位：m
    # dy = dx  # y方向单元间距，单位：m
    # k = 2 * np.pi / lamda0  # 波数
    beamvec = [sin(radians(theta0)) * cos(radians(phi0)),
               sin(radians(theta0)) * sin(radians(phi0)), 0]

    # 生成天线位置坐标
    posx = np.arange(-dx * (nRow - 1) / 2, dx * (nRow - 1) / 2 + dx, dx)
    posy = np.arange(-dy * (mLine - 1) / 2, dy * (mLine - 1) / 2 + dy, dy)

    # 参考阵元位置
    Rc = [posx[0], posy[0], 0]

    # # 配置馈元和计算馈元位置
    # InF = 1  # 喇叭馈源摆放状态，0：垂直阵面，1：指向阵面中心
    # InTetha = 30  # 入射角
    # aphea_10dB = 45  # 喇叭馈源的-10dB波束角宽度
    Feed, FD = get_feed(InF, InTetha, aphea_10dB, nRow, dx)

    # 计算反射面相位分布
    phase = get_phase(posx, posy, beamvec, Feed, Rc, k)

    # 将相位从弧度转角度
    # phase = np.rad2deg(phase)

    return phase