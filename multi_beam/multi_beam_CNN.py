import os
import logging
import h5py
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.models import load_model

from util.util_log import setup_logging
from util.util_ris_pattern import phase_2_pattern, point_2_phi_pattern, phase_2_pattern_xyz
from util.util_csv import save_csv
from util.util_image import save_img, save_img_xyz
from util.util_h5 import read_h5


# ============================================= CNN 数据集生成 =====================================================
# 生成数据集波束指向角列表
def get_theta0s_phi0s(theta_min, theta_max, theta_step, phi_min, phi_max, phi_step):
    ang_list = []
    for theta in range(theta_min, theta_max+1, theta_step):
        for phi in range(phi_min, phi_max + 1, phi_step):
            ang_list.append([theta, phi])
    return ang_list


# 批量保存h5
def save_h5_in_chunks(path_h5, X, Y):
    # 假设 X 是一个形状为 (60000, 500, 500) 的numpy数组，Y 是形状为 (60000, 8, 8) 的numpy数组
    logger.info(f"[CNN dataset gen] 保存数据到文件{path_h5}")
    with h5py.File(path_h5, 'a') as hf:
        # 检查 'inputs' 数据集是否存在
        if 'inputs' in hf:
            logger.info(f"[CNN dataset gen] 保存数据--添加input到文件{path_h5}")
            # 获取当前 'inputs' 数据集的大小
            current_size = hf['inputs'].shape[0]
            # 扩展数据集大小以容纳新数据
            hf['inputs'].resize((current_size + X.shape[0], *X.shape[1:]))
            # 追加新数据到 'inputs'
            hf['inputs'][current_size:] = X
        else:
            logger.info(f"[CNN dataset gen] 保存数据--新建input到文件{path_h5}")
            # 如果 'inputs' 不存在，首次创建时指定maxshape为None
            hf.create_dataset('inputs', data=X, maxshape=(None, *X.shape[1:]), chunks=True, compression="gzip", compression_opts=9)

        # 对 'outputs' 数据集执行相同的操作
        if 'outputs' in hf:
            logger.info(f"[CNN dataset gen] 保存数据--添加output到文件{path_h5}")
            current_size = hf['outputs'].shape[0]
            hf['outputs'].resize((current_size + Y.shape[0], *Y.shape[1:]))
            hf['outputs'][current_size:] = Y
        else:
            logger.info(f"[CNN dataset gen] 保存数据--新建output到文件{path_h5}")
            # 首次创建 'outputs' 时同样指定maxshape为None
            hf.create_dataset('outputs', data=Y, maxshape=(None, *Y.shape[1:]), chunks=True, compression="gzip", compression_opts=9)


# 生成数据集主函数
def main_gen_dataset(theta0_start, theta0_end, theta0_step, phi0_start, phi0_end, phi0_step,
                     path_h5_pre, path_img_pre, start_index=1, batch_size=1000):
    logger.info("[CNN dataset gen]theta0_start=%f,theta0_end=%f,theta0_step=%f,phi0_start=%f,phi0_end=%f,phi0_step=%f" % (
        theta0_start, theta0_end, theta0_step, phi0_start, phi0_end, phi0_step))
    #
    path_img_pattern_pre = path_img_pre + "pattern/"
    path_img_phaseBit_pre = path_img_pre + "phaseBit/"
    path_h5_name = "(" + str(theta0_start) + "," + str(theta0_end) + "," + str(theta0_step) \
                   + "," + str(phi0_start) + "," + str(phi0_end) + "," + str(phi0_step) + ")"
    path_h5_bit = path_h5_pre + "single-phase-pattern-64x64-"+ path_h5_name +".h5"
    path_h5_point = path_h5_pre + "single-phase-point-64x64-300_point.h5"
    # 生成扫描指向角列表
    ang_list = get_theta0s_phi0s(theta0_start, theta0_end, theta0_step, phi0_start, phi0_end, phi0_step)
    logger.info("[CNN dataset gen]len of ang_list: %d" % len(ang_list))
    # 分批保存
    for i in range(start_index, len(ang_list), batch_size):
        end_index = min(i + batch_size, len(ang_list))
        logger.info(f'[CNN dataset gen] 正在处理第{i}至{end_index - 1}条数据，共{len(ang_list)}条。')
        # 单批次处理
        batch_ang = ang_list[i:end_index]
        batch_phaseBit = []
        batch_pattern = []
        batch_point = []
        for j in range(len(batch_ang)):
            theta0 = batch_ang[j][0]
            phi0 = batch_ang[j][1]
            logger.info(f'[CNN dataset gen] 迭代到第{j} 次，共{len(batch_ang)}次。theta0: {theta0}, phi0: {phi0}')
            # 计算方向图
            phase, phaseBit, pattern = point_2_phi_pattern(theta0, phi0)
            # 保存
            batch_phaseBit.append(phaseBit)
            batch_pattern.append(pattern)
            batch_point.append(np.array(batch_ang[j]))
            # 画图
            # path_img_phaseBit = path_img_phaseBit_pre + "(" + str(theta0) + "," + str(phi0) + ").jpg"
            # path_img_pattern = path_img_pattern_pre + "(" + str(theta0) + "," + str(phi0) + ").jpg"
            # logger.info(f'path_img_phaseBit: {path_img_phaseBit}, path_img_pattern: {path_img_pattern}')
            # save_img(path_img_phaseBit, phaseBit)
            # save_img(path_img_pattern, pattern)
        # 记录到h5
        np_batch_phaseBit = np.array(batch_phaseBit)
        np_batch_pattern = np.array(batch_pattern)
        np_batch_point = np.array(batch_point)
        save_h5_in_chunks(path_h5_bit, np_batch_phaseBit, np_batch_pattern)
        save_h5_in_chunks(path_h5_point, np_batch_phaseBit, np_batch_point)


# ============================================= CNN 相关 =====================================================
class CNN1():
    path_datas = [
        "../../files/dataset/point/singel-64-64-(20,40,1,-15,15,1)/single-phase-pattern-64x64-(20,40,1,-15,15,1).h5",
        "../../files/dataset/point/singel-64-64-(20,40,1,75,105,1)/single-phase-pattern-64x64-(20,40,1,75,105,1).h5",
        "../../files/dataset/point/singel-64-64-(20,40,1,165,195,1)/single-phase-pattern-64x64-(20,40,1,165,195,1).h5",
        "../../files/dataset/point/singel-64-64-(20,40,1,255,285,1)/single-phase-pattern-64x64-(20,40,1,255,285,1).h5"]
    path_model = "../../files/multi-beam/1bit/CNN/model-1bit-single-64x64-cnn1-(20,40,1,0-90-180-270).h5"

    def train(self):
        # 读取.h5py
        inputs_list = []
        outputs_list = []
        for path_h5 in self.path_datas:
            inputs, outputs = read_h5(path_h5)
            inputs_list.append(inputs)
            outputs_list.append(outputs)
        X_train = np.concatenate(outputs_list, axis=0)
        Y_train = np.concatenate(inputs_list, axis=0)
        # 打印分割后的数据集大小以确认
        logger.info("[CNN train] X_train shape: %s" % str(X_train.shape))
        logger.info("[CNN train] Y_train shape: %s" % str(Y_train.shape))
        # 设置尺寸
        pattern_x = X_train.shape[1]
        pattern_y = X_train.shape[2]
        phi_x = Y_train.shape[1]
        phi_y = Y_train.shape[2]
        logger.info("[CNN train] pattern_x:%d, pattern_y:%d, phi_x:%d, phi_y:%d" % (pattern_x, pattern_y, phi_x, phi_y))
        # 创建模型
        model = Sequential([
            # 添加卷积层和池化层
            Conv2D(64, (3, 3), activation='relu', input_shape=(pattern_x, pattern_y, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(512, (3, 3), activation='relu'),  # 增加卷积层
            MaxPooling2D((2, 2)),
            Conv2D(512, (3, 3), activation='relu'),  # 增加卷积层
            MaxPooling2D((2, 2)),
            # 扁平化输出用于全连接层
            Flatten(),
            # 添加全连接层，使用较少的节点逐渐减少维度
            Dense(1024, activation='relu'),  # 增加全连接层的节点数
            Dense(512, activation='relu'),  # 增加全连接层的节点数
            Dense(phi_x * phi_y, activation='linear'),  # 输出层，没有激活函数，6x6=36
            # 调整输出大小符合输出的二维数组
            Reshape((phi_x, phi_y)),
        ])
        # 编译模型，使用均方误差作为损失函数，适合回归任务
        model.compile(optimizer='adam', loss='mean_squared_error')
        # 训练模型并记录历史
        history = model.fit(X_train, Y_train, epochs=20, batch_size=8, verbose=1)
        # 打印损失值的变化
        logger.info("[CNN train] Loss per epoch: %s" % str(history.history['loss']))
        # 预测测试集
        # 训练完成后保存模型
        model.save(self.path_model)
        logger.info("[CNN train] Model saved to disk. path = %s" % (self.path_model))

    def test(self, pattern):
        # 加载预先训练好的模型
        if os.path.exists(self.path_model):
            model = load_model(self.path_model)
            logger.info("[CNN test] Model loaded from disk: %s" % (self.path_model))
        else:
            logger.error("[CNN test] not found: %s" % (self.path_model))
            raise FileNotFoundError(f"Model file {self.path_model} not found.")
        # 使用模型进行预测
        logger.info("[CNN test] predict pattern.")
        # 将pattern变形为期望的形状(None, 360, 361, 1)
        pattern_reshaped = pattern.reshape(1, 360, 361, 1)
        # 使用变形后的pattern进行预测
        phase = model.predict(pattern_reshaped)
        # phase = model.predict(pattern)
        return phase


class CNN2():
    path_datas = ["../../files/dataset/point/singel-64-64-(20,40,1,-15,15,1)/single-phase-pattern-64x64-(20,40,1,-15,15,1).h5",
                  "../../files/dataset/point/singel-64-64-(20,40,1,75,105,1)/single-phase-pattern-64x64-(20,40,1,75,105,1).h5",
                  "../../files/dataset/point/singel-64-64-(20,40,1,165,195,1)/single-phase-pattern-64x64-(20,40,1,165,195,1).h5",
                  "../../files/dataset/point/singel-64-64-(20,40,1,255,285,1)/single-phase-pattern-64x64-(20,40,1,255,285,1).h5"]
    path_model = "../../files/multi-beam/1bit/CNN/model-1bit-single-64x64-cnn2-(20,40,1,0-90-180-270).h5"

    def train(self):
        # 读取.h5py
        inputs_list = []
        outputs_list = []
        for path_h5 in self.path_datas:
            inputs, outputs = read_h5(path_h5)
            inputs_list.append(inputs)
            outputs_list.append(outputs)
        X_train = np.concatenate(outputs_list, axis=0)
        Y_train = np.concatenate(inputs_list, axis=0)
        # 打印分割后的数据集大小以确认
        logger.info("[CNN2 train] X_train shape: %s" % str(X_train.shape))
        logger.info("[CNN2 train] Y_train shape: %s" % str(Y_train.shape))
        # 设置尺寸
        pattern_x = X_train.shape[1]
        pattern_y = X_train.shape[2]
        phi_x = Y_train.shape[1]
        phi_y = Y_train.shape[2]
        logger.info("[CNN2 train] pattern_x:%d, pattern_y:%d, phi_x:%d, phi_y:%d" % (pattern_x, pattern_y, phi_x, phi_y))
        # 创建模型
        model = Sequential([
            # 添加卷积层和池化层
            Conv2D(64, (2, 2), activation='relu', input_shape=(pattern_x, pattern_y, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(128, (2, 2), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (2, 2), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(512, (2, 2), activation='relu'),  # 增加卷积层
            MaxPooling2D((2, 2)),
            Conv2D(512, (2, 2), activation='relu'),  # 增加卷积层
            MaxPooling2D((2, 2)),
            # 新增卷积层
            Conv2D(1024, (2, 2), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(1024, (2, 2), activation='relu'),
            MaxPooling2D((2, 2)),
            # 扁平化输出用于全连接层
            Flatten(),
            # 添加全连接层，使用较少的节点逐渐减少维度
            Dense(1024, activation='relu'),  # 增加全连接层的节点数
            Dense(512, activation='relu'),  # 增加全连接层的节点数
            Dense(phi_x * phi_y, activation='linear'),  # 输出层，没有激活函数，6x6=36
            # 调整输出大小符合输出的二维数组
            Reshape((phi_x, phi_y)),
        ])
        # 编译模型，使用均方误差作为损失函数，适合回归任务
        model.compile(optimizer='adam', loss='mean_squared_error')
        # 训练模型并记录历史
        history = model.fit(X_train, Y_train, epochs=20, batch_size=8, verbose=1)
        # 打印损失值的变化
        logger.info("[CNN2 train] Loss per epoch: %s" % str(history.history['loss']))
        # 预测测试集
        # 训练完成后保存模型
        model.save(self.path_model)
        logger.info("[CNN2 train] Model saved to disk. path = %s" % (self.path_model))

    def test(self, pattern):
        # 加载预先训练好的模型
        if os.path.exists(self.path_model):
            model = load_model(self.path_model)
            logger.info("[CNN2 test] Model loaded from disk: %s" % (self.path_model))
        else:
            logger.error("[CNN2 test] not found: %s" % (self.path_model))
            raise FileNotFoundError(f"[CNN2 test] Model file {self.path_model} not found.")
        # 使用模型进行预测
        logger.info("[CNN2 test] predict pattern.")
        # 将pattern变形为期望的形状(None, 360, 361, 1)
        pattern_reshaped = pattern.reshape(1, 360, 361, 1)
        # 使用变形后的pattern进行预测
        phase = model.predict(pattern_reshaped)
        # phase = model.predict(pattern)
        return phase

# ============================================= 主函数 ========================================================
# 根据波束指向生成方向图
def create_cnn_x_2(theta1, phi1, theta2, phi2):
    # 计算两个指向的方向图
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2)
    # pattern_mix选择pattern1和pattern2中更大的那个
    pattern_mix = np.maximum(pattern1, pattern2)
    return pattern_mix

# 根据波束指向生成方向图
def create_cnn_x_4(theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4):
    # 计算两个指向的方向图
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2)
    phase3, phaseBit3, pattern3 = point_2_phi_pattern(theta3, phi3)
    phase4, phaseBit4, pattern4 = point_2_phi_pattern(theta4, phi4)
    # pattern_mix选择patternx中最大的那个
    pattern_mix  =  np.maximum(pattern1,  np.maximum(pattern2,  np.maximum(pattern3,  pattern4)))
    return pattern_mix

# 训练网络
def main_train():
    logger.info("[CNN train] CNN1 train start.")
    cnn1 = CNN1()
    cnn1.train()
    logger.info("[CNN train] CNN1 train finish. And CNN2 train start.")
    cnn2 = CNN2()
    cnn2.train()
    logger.info("[CNN train] CNN2 train finish.")

# 基于CNN的方法 -- 双波束
def main_multi_beam_2(theta1, phi1, theta2, phi2, path_pre):
    logger.info("main_multi_beam_2: theta1=%d, phi1=%d, theta2=%d, phi2=%d, " % (theta1, phi1, theta2, phi2))
    # 根据波束指向生成方向图
    pattern_mix_cnn_x = create_cnn_x_2(theta1, phi1, theta2, phi2)
    # 使用cnn计算码阵
    cnn = CNN1()
    phase_cnn_y = cnn.test(pattern_mix_cnn_x)
    print("phase_cnn_y.shape:", phase_cnn_y.shape)
    phase_cnn_y = np.squeeze(phase_cnn_y)
    print("phase_cnn_y.shape:", phase_cnn_y.shape)
    # 相位转换1bit -- 1bit
    phaseBit_cnn_y = np.where(phase_cnn_y >= 0.5, 1, 0)
    print("phaseBit_cnn_y.shape:", phaseBit_cnn_y.shape)
    # 计算phase_mix的方向图
    phase_cnn_y = phase_cnn_y * 180
    phaseBit_cnn_y = phaseBit_cnn_y * 180
    pattern_mix = phase_2_pattern(phase_cnn_y)
    patternBit_mix = phase_2_pattern(phaseBit_cnn_y)
    pattern_mix_xyz, x_mix, y_mix, z_mix = phase_2_pattern_xyz(phase_cnn_y)
    patternBit_mix_xyz, x_bit_mix, y_bit_mix, z_bit_mix = phase_2_pattern_xyz(phaseBit_cnn_y)
    #
    # 保存结果
    logger.info("save CNN multi-beam 2 result...")
    # 保存图片
    save_img(path_pre + "phase_mix.jpg", phase_cnn_y)
    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_cnn_y)  # CNN法 -- 结果码阵
    save_img(path_pre + "pattern_mix.jpg", pattern_mix)
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)     # CNN法 -- 结果码阵方向图
    save_img(path_pre + "pattern_mix_cnn_x.jpg", pattern_mix_cnn_x)
    save_img_xyz(path_pre + "pattern_mix_xyz.jpg", np.abs(pattern_mix_xyz), x_mix, y_mix)
    save_img_xyz(path_pre + "patternBit_mix_xyz.jpg", np.abs(patternBit_mix_xyz), x_bit_mix, y_bit_mix)
    # 保存相位结果
    save_csv(phase_cnn_y, path_pre + "phase_mix.csv")
    save_csv(phaseBit_cnn_y, path_pre + "phaseBit_mix.csv")


# 基于CNN的方法 -- 四波束
def main_multi_beam_4(theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4, path_pre):
    logger.info("main_multi_beam_2: theta1=%d, phi1=%d, theta2=%d, phi2=%d, theta3=%d, phi3=%d, theta4=%d, phi4=%d"
                % (theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4))
    # 根据波束指向生成方向图
    pattern_mix_cnn_x = create_cnn_x_4(theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4)
    # 使用cnn计算码阵
    cnn = CNN1()
    phase_cnn_y = cnn.test(pattern_mix_cnn_x)
    print("phase_cnn_y.shape:", phase_cnn_y.shape)
    phase_cnn_y = np.squeeze(phase_cnn_y)
    print("phase_cnn_y.shape:", phase_cnn_y.shape)
    # 相位转换1bit -- 1bit
    phaseBit_cnn_y = np.where(phase_cnn_y >= 0.5, 1, 0)
    print("phaseBit_cnn_y.shape:", phaseBit_cnn_y.shape)
    # 计算phase_mix的方向图
    pattern_mix = phase_2_pattern(phase_cnn_y)
    patternBit_mix = phase_2_pattern(phaseBit_cnn_y)
    #
    # 保存结果
    logger.info("save CNN multi-beam 4 result...")
    # 保存图片
    save_img(path_pre + "phase_mix.jpg", phase_cnn_y)
    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_cnn_y)  # CNN法 -- 结果码阵
    save_img(path_pre + "pattern_mix.jpg", pattern_mix)
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)     # CNN法 -- 结果码阵方向图
    save_img(path_pre + "pattern_mix_cnn_x.jpg", pattern_mix_cnn_x)
    # 保存相位结果
    save_csv(phase_cnn_y, path_pre + "phase_mix.csv")
    save_csv(phaseBit_cnn_y, path_pre + "phaseBit_mix.csv")


if __name__ == '__main__':
    # 配置日志，默认打印到控制台，也可以设置打印到文件
    setup_logging()
    # setup_logging(log_file="../../files/logs/log_multi_beam_CNN.log")

    # 获取日志记录器并记录日志
    logger = logging.getLogger("[RIS-multi-beam-CNN-1bit]")
    logger.info("1bit-RIS-multi-beam-CNN: CNN based method")

    # 基于CNN的方法: 主函数
    # 0. 生成数据集主函数
    # main_gen_dataset(20, 40, 1, -15, 15, 1,
    #                  "../../files/dataset/point/singel-64-64-(20,40,1,-15,15,1)/",
    #                  "../../files/dataset/point/singel-64-64-(20,40,1,-15,15,1)/image/")
    # main_gen_dataset(20, 40, 1, 75, 105, 1,
    #                  "../../files/dataset/point/singel-64-64-(20,40,1,75,105,1)/",
    #                  "../../files/dataset/point/singel-64-64-(20,40,1,75,105,1)/image/")
    # main_gen_dataset(20, 40, 1, 165, 195, 1,
    #                  "../../files/dataset/point/singel-64-64-(20,40,1,165,195,1)/",
    #                  "../../files/dataset/point/singel-64-64-(20,40,1,165,195,1)/image/")
    # main_gen_dataset(20, 40, 1, 255, 285, 1,
    #                  "../../files/dataset/point/singel-64-64-(20,40,1,255,285,1)/",
    #                  "../../files/dataset/point/singel-64-64-(20,40,1,255,285,1)/image/")
    # 1. 训练主函数
    # main_train()
    # 2. 测试主函数
    main_multi_beam_2(30, 0, 30, 90, "../../files/multi-beam/1bit/CNN/2-(30,0,30,90)/CNN1-20-8/")
    # main_multi_beam_2(30, 0, 30, 180, "../../files/multi-beam/1bit/CNN/2-(30,0,30,180)/")
    # main_multi_beam_4(30, 0, 30, 90, 30, 180, 30, 270, "../../files/multi-beam/CNN/4-(30,0,30,90,30,180,30,270)/")