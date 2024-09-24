import sys
import logging
import random
import numpy as np

from util.util_log import setup_logging
from util.util_csv import save_csv
from util.util_image import save_img, draw_img
from util.util_ris_pattern import point_2_phi_pattern, parse_pattern_dbw, phase_2_pattern

from multi_beam_PS import psm_beam_2, psm_beam_4


# ============================================= GA multi-beam ========================================================
import random

class GeneticAlgorithm:
    def __init__(self, fitness_func, gene_set, population_size, chromosome_length, mutation_rate, crossover_rate):
        self.fitness_func = fitness_func
        self.gene_set = gene_set
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        return [self.generate_chromosome() for _ in range(self.population_size)]

    def generate_chromosome(self):
        return [random.choice(self.gene_set) for _ in range(self.chromosome_length)]

    def evaluate_fitness(self):
        return [self.fitness_func(chromosome) for chromosome in self.population]

    def select_parents(self, fitness_values):
        total_fitness = sum(fitness_values)
        selection_probs = [fitness / total_fitness for fitness in fitness_values]
        parents = random.choices(self.population, weights=selection_probs, k=2)
        return parents

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.chromosome_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2

    def mutate(self, chromosome):
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.choice(self.gene_set)
        return chromosome

    def evolve(self):
        fitness_values = self.evaluate_fitness()
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(fitness_values)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.extend([child1, child2])
        self.population = new_population[:self.population_size]

    def run(self, generations):
        for _ in range(generations):
            self.evolve()
        best_chromosome = max(self.population, key=self.fitness_func)
        return best_chromosome


def testGA():
    def fitness_function(chromosome):
        target = "Hello, World!"
        score = sum(1 for i, j in zip(chromosome, target) if i == j)
        return score

    gene_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ, !"
    population_size = 500
    chromosome_length = len("Hello, World!")
    mutation_rate = 0.01
    crossover_rate = 0.8
    generations = 100

    ga = GeneticAlgorithm(fitness_function, gene_set, population_size, chromosome_length, mutation_rate, crossover_rate)
    best_chromosome = ga.run(generations)
    print(''.join(best_chromosome))


class GA():
    # 参数 -- 1.天线参数
    N_theta = 361   # theta方向采样数
    N_phi = 360     # phi方向采样数

    # 参数 -- 2.GA参数
    POP_SIZE = 2           # 种群大小
    MAX_GENERATIONS = 1   # 最大代数
    CROSSOVER_RATE = 0.7    # 交叉率
    MUTATION_RATE = 0.05    # 变异率

    # 参数 -- 3.GA-fitness-阈值
    MASK_UP_WIDTH_HALF = 5
    MASK_UP_VAL_HIGH = 0
    MASK_UP_VAL_LOW = -15
    MASK_DOWN_WIDTH_HALF = 2
    MASK_DOWN_VAL_HIGH = -3
    MASK_DOWN_VAL_LOW = -40

    mask_up = None
    mask_down = None

    #
    BEST_MATCH_VAL = None               # 最佳适应度矩阵
    BEST_MATCH_FITNESS = sys.maxsize    # 最佳适应度值
    BEST_MATCH_FITNESS_LIST = []        # 最佳适应度变化列表, 用于画进化曲线

    # 根据指向角生成掩码 -- 双波束
    # def __init_mask_2__(self, theta1, phi1, theta2, phi2):
    #     # 初始化mask_up和mask_down数组
    #     self.mask_up = np.full((self.N_theta, self.N_phi), self.MASK_UP_VAL_LOW)
    #     self.mask_down = np.full((self.N_theta, self.N_phi), self.MASK_DOWN_VAL_LOW)
    #     # 计算中心点的索引
    #     center1 = (int(theta1 * self.N_theta / 90), int(phi1 * self.N_phi / 360))
    #     center2 = (int(theta2 * self.N_theta / 90), int(phi2 * self.N_phi / 360))
    #
    #     # # 定义一个函数来更新mask中的值
    #     # def update_mask(mask, center, width, high_val):
    #     #     for i in range(max(0, center[0] - width), min(self.N_theta, center[0] + width + 1)):
    #     #         for j in range(max(0, center[1] - width), min(self.N_phi, center[1] + width + 1)):
    #     #             if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= width ** 2:
    #     #                 mask[i, j] = high_val
    #     # 定义一个函数来更新mask中的值，考虑边界环绕
    #     def update_mask(mask, center, width, high_val, N_theta, N_phi):
    #         for i in range(center[0] - width, center[0] + width + 1):
    #             for j in range(center[1] - width, center[1] + width + 1):
    #                 if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= width ** 2:
    #                     # 处理环绕边界
    #                     ii = i % N_theta
    #                     jj = j % N_phi
    #                     mask[ii, jj] = high_val
    #
    #     # 更新mask_up和mask_down
    #     update_mask(self.mask_up, center1, self.MASK_UP_WIDTH_HALF, self.MASK_UP_VAL_HIGH, self.N_theta, self.N_phi)
    #     update_mask(self.mask_up, center2, self.MASK_UP_WIDTH_HALF, self.MASK_UP_VAL_HIGH, self.N_theta, self.N_phi)
    #     update_mask(self.mask_down, center1, self.MASK_DOWN_WIDTH_HALF, self.MASK_DOWN_VAL_HIGH, self.N_theta,
    #                 self.N_phi)
    #     update_mask(self.mask_down, center2, self.MASK_DOWN_WIDTH_HALF, self.MASK_DOWN_VAL_HIGH, self.N_theta,
    #                 self.N_phi)
    def __init_mask_2__(self, theta1, phi1, theta2, phi2):
        # 初始化mask_up和mask_down数组
        self.mask_up = np.full((self.N_phi, self.N_theta), self.MASK_UP_VAL_LOW)
        self.mask_down = np.full((self.N_phi, self.N_theta), self.MASK_DOWN_VAL_LOW)
        # 计算中心点的索引
        center1 = (int(phi1 * self.N_phi / 360), int(theta1 * self.N_theta / 90))
        center2 = (int(phi2 * self.N_phi / 360), int(theta2 * self.N_theta / 90))

        def update_mask(mask, center, width, high_val, N_theta, N_phi):
            for i in range(center[0] - width, center[0] + width + 1):
                for j in range(center[1] - width, center[1] + width + 1):
                    if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= width ** 2:
                        # 处理环绕边界
                        ii = i % N_phi
                        jj = j % N_theta
                        mask[ii, jj] = high_val

        # 更新mask_up和mask_down
        update_mask(self.mask_up, center1, self.MASK_UP_WIDTH_HALF, self.MASK_UP_VAL_HIGH, self.N_theta, self.N_phi)
        update_mask(self.mask_up, center2, self.MASK_UP_WIDTH_HALF, self.MASK_UP_VAL_HIGH, self.N_theta, self.N_phi)
        update_mask(self.mask_down, center1, self.MASK_DOWN_WIDTH_HALF, self.MASK_DOWN_VAL_HIGH, self.N_theta,
                    self.N_phi)
        update_mask(self.mask_down, center2, self.MASK_DOWN_WIDTH_HALF, self.MASK_DOWN_VAL_HIGH, self.N_theta,
                    self.N_phi)

    # 根据指向角生成掩码 -- 四波束
    def __init_mask_4__(self, theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4):
        # 初始化mask_up和mask_down数组
        self.mask_up = np.full((self.N_phi, self.N_theta), self.MASK_UP_VAL_LOW)
        self.mask_down = np.full((self.N_phi, self.N_theta), self.MASK_DOWN_VAL_LOW)
        # 计算中心点的索引
        centers = [(int(phi1 * self.N_phi / 360), int(theta1 * self.N_theta / 90)),
                   (int(phi2 * self.N_phi / 360), int(theta2 * self.N_theta / 90)),
                   (int(phi3 * self.N_phi / 360), int(theta3 * self.N_theta / 90)),
                   (int(phi4 * self.N_phi / 360), int(theta4 * self.N_theta / 90))]

        def update_mask(mask, centers, width, high_val, N_theta, N_phi):
            for center in centers:
                for i in range(center[0] - width, center[0] + width + 1):
                    for j in range(center[1] - width, center[1] + width + 1):
                        if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= width ** 2:
                            # 处理环绕边界
                            ii = i % N_phi
                            jj = j % N_theta
                            mask[ii, jj] = high_val

        # 更新mask_up和mask_down
        update_mask(self.mask_up, centers, self.MASK_UP_WIDTH_HALF, self.MASK_UP_VAL_HIGH, self.N_theta, self.N_phi)
        update_mask(self.mask_down, centers, self.MASK_DOWN_WIDTH_HALF, self.MASK_DOWN_VAL_HIGH, self.N_theta, self.N_phi)

    # 计算适应度
    def __fitness__(self, phase):
        pattern_dbw = parse_pattern_dbw(phase)
        # 初始化误差为0
        err = 0.0
        # 获取数组的形状
        M, N = pattern_dbw.shape
        # 遍历数组中的每个元素
        for i in range(M):
            for j in range(N):
                # 检查pattern[i][j]是否在mask_up[i][j]和mask_down[i][j]之间
                if pattern_dbw[i][j] > self.mask_up[i][j]:
                    # 超出mask_up的误差平方
                    err += (pattern_dbw[i][j] - self.mask_up[i][j]) ** 2
                elif pattern_dbw[i][j] < self.mask_down[i][j]:
                    # 超出mask_down的误差平方
                    err += (self.mask_down[i][j] - pattern_dbw[i][j]) ** 2
        return err

    # 选择过程, 计算概率
    def __get_choose_weights(self, population):
        sequence = []
        fitness_max = 0
        for individual in population:
            fitness = self.__fitness__(individual)
            if fitness_max < fitness:
                fitness_max = fitness
            sequence.append(fitness)
        # print("sequence:", sequence)
        # print("fitness_max:", fitness_max)
        for i in range(0, len(sequence)):
            sequence[i] = np.abs(fitness_max + 1 - sequence[i])/fitness_max * 100
        # print("sequence new:", sequence)
        return sequence
    # 选择过程, 基于概率
    def __choose__(self, population):
        # return random.choices(
        #     population,
        #     weights=[self.__fitness__(individual) for individual in population],
        #     k=self.POP_SIZE)
        logger.info("[GA] choose: get_choose_weights")
        sequence = self.__get_choose_weights(population)
        choices = random.choices(population, weights=sequence, k=self.POP_SIZE)
        res_list = []
        for choice in choices:
            res_list.append(np.array(choice))
        return res_list

    # 交叉操作, 基于概率
    def __crossover__(self, a, b):
        # 展平二维数组
        a_flat = a.flatten()
        b_flat = b.flatten()
        # 交叉操作
        M, N = a.shape
        if random.random() < self.CROSSOVER_RATE:
            point = random.randint(1, M * N - 1)
            # 交叉操作
            offspring_flat = np.concatenate((a_flat[:point], b_flat[point:]))
        else:
            # 如果没有进行交叉，选择其中一个作为后代
            offspring_flat = a_flat if random.random() < 0.5 else b_flat
        # 将结果复原为二维数组
        offspring = offspring_flat.reshape(M, N)
        return offspring

    # 变异操作, 基于概率
    def __mutate__(self, value):
        value = list(value)
        for i in range(len(value)):
            for j in range(len(value[0])):
                if random.random() < self.MUTATION_RATE:
                    value[i][j] = random.randint(0, 1)
        return value

    def run(self, phaseBit_mix_init):
        # 清除适应度变化列表
        self.BEST_MATCH_FITNESS_LIST.clear()
        # 使用 phaseBit_mix_init, 创建初始种群
        logger.info("[GA-AP] generate init pop.")
        population = [phaseBit_mix_init for _ in range(self.POP_SIZE)]  # for _ in range就是创建循环
        # 运行遗传算法
        generation = 0
        # best_match = population[1]
        logger.info("[GA-AP] search begin")
        for generation in range(self.MAX_GENERATIONS):
            # 评估和选择
            logger.info("[GA-AP] __choose__")
            population = self.__choose__(population)
            # 产生后代
            new_population = []  # 创建空队列newpopulation
            for i in range(0, self.POP_SIZE, 2):
                parent1 = population[i]
                parent2 = population[i + 1]
                child1 = self.__crossover__(parent1, parent2)
                child2 = self.__crossover__(parent2, parent1)
                new_population.extend([self.__mutate__(child1), self.__mutate__(child2)])  # 给newpopulation列表添加新元素
            logger.info("[GA-AP] new_population")
            population = new_population
            # 打印最佳的匹配结果
            # best_match = min(population, key=self.__fitness__)
            # logger.info(f"Generation {generation}: {best_match} (Fitness: {self.__fitness__(best_match)})")
            logger.info("[GA-AP] find best_match_fit")
            best_match_fit = self.__fitness__(population[0])
            best_match_p = population[0]
            for p in population:
                p_fitness = self.__fitness__(p)
                if p_fitness < best_match_fit:
                    best_match_fit = p_fitness
                    best_match_p = p
            # logger.info(f"Generation {generation}: (Fitness: {best_match_p}), best fitness: {self.BEST_MATCH_FITNESS}")
            logger.info(f"Generation {generation}: best fitness: {self.BEST_MATCH_FITNESS}")
            if best_match_fit < self.BEST_MATCH_FITNESS:
                self.BEST_MATCH_FITNESS = best_match_fit
                self.BEST_MATCH_VAL = best_match_p
                self.BEST_MATCH_FITNESS_LIST.append(best_match_fit)
            else:
                self.BEST_MATCH_FITNESS_LIST.append(self.BEST_MATCH_FITNESS)
            # 如果找到准确的匹配，则停止
            if best_match_fit == 0:
                self.BEST_MATCH_FITNESS = best_match_fit
                self.BEST_MATCH_VAL = best_match_p
                break
        logger.info(f"Target matched after {generation} generations!")

    # 核心方法: 遗传算法双波束扫描
    def beamScanning2Dual(self, theta1, phi1, theta2, phi2, phaseBit_mix_init):
        # 根据 交替投影法 生成上下掩码
        logger.info("[GA-AP] generate mask_up & mask_down")
        self.__init_mask_2__(theta1, phi1, theta2, phi2)
        # 运行GA
        self.run(phaseBit_mix_init)
        #
        return self.BEST_MATCH_VAL, self.BEST_MATCH_FITNESS, self.BEST_MATCH_FITNESS_LIST

    # 核心方法: 遗传算法双波束扫描
    def beamScanning4Dual(self, theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4, phaseBit_mix_init):
        # 根据 交替投影法 生成上下掩码
        logger.info("[GA-AP] generate mask_up & mask_down")
        self.__init_mask_4__(theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4)
        # 运行GA
        self.run(phaseBit_mix_init)
        #
        return self.BEST_MATCH_VAL, self.BEST_MATCH_FITNESS, self.BEST_MATCH_FITNESS_LIST


# ============================================= 主函数 ========================================================
# 基于GA(寻优方法) - 交替投影(GA的fitness) - 相位叠加(初始位置) 的方法 -- 双波束
def main_multi_beam_2(theta1, phi1, theta2, phi2):
    logger.info("main_multi_beam_2: theta1=%d, phi1=%d, theta2=%d, phi2=%d, " % (theta1, phi1, theta2, phi2))
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2)
    # GA-AP算法
    # 使用 相位叠加法(PS) 得到初始位置
    logger.info("[STEP 1] use PS to get init phase.")
    phase_mix_ps, phaseBit_mix_ps = psm_beam_2(phase1, phase2)
    # 使用 交替投影法 设置GA的fitness
    logger.info("[STEP 2] use GA-AP to get optimized phase.")
    ga = GA()
    phaseBit_mix, fitness, fitness_list = ga.beamScanning2Dual(theta1, phi1, theta2, phi2, phaseBit_mix_ps)
    logger.info("fitness: %f" % (fitness))
    # 计算phase_mix的方向图
    phaseBit_mix = np.array(phaseBit_mix)
    patternBit_mix = phase_2_pattern(phaseBit_mix)
    #
    # 保存结果
    logger.info("save GA-AP multi-beam 2 result...")
    path_pre = "../../files/multi-beam/GA-AP/2-(30,0,30,180)/"
    # 保存图片
    save_img(path_pre + "phase1.jpg", phase1)
    save_img(path_pre + "phase2.jpg", phase2)
    save_img(path_pre + "phaseBit1.jpg", phaseBit1)
    save_img(path_pre + "phaseBit2.jpg", phaseBit2)
    save_img(path_pre + "pattern1.jpg", pattern1)
    save_img(path_pre + "pattern2.jpg", pattern2)
    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_mix)         # GA法 -- 结果码阵
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)     # GA法 -- 结果码阵方向图
    # 保存相位结果
    save_csv(phase1, path_pre + "phase1.csv")
    save_csv(phase2, path_pre + "phase2.csv")
    save_csv(phaseBit1, path_pre + "phaseBit1.csv")
    save_csv(phaseBit2, path_pre + "phaseBit2.csv")
    save_csv(phaseBit_mix, path_pre + "phaseBit_mix.csv")
    #fitness_list = [float(x) for x in fitness_list]
    #save_csv(fitness_list, path_pre + "fitness.csv")


# 基于GA(寻优方法) - 交替投影(GA的fitness) - 相位叠加(初始位置) 的方法 -- 四波束
def main_multi_beam_4(theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4):
    logger.info("main_multi_beam_2: theta1=%d, phi1=%d, theta2=%d, phi2=%d, theta3=%d, phi3=%d, theta4=%d, phi4=%d"
                % (theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4))
    phase1, phaseBit1, pattern1 = point_2_phi_pattern(theta1, phi1)
    phase2, phaseBit2, pattern2 = point_2_phi_pattern(theta2, phi2)
    phase3, phaseBit3, pattern3 = point_2_phi_pattern(theta3, phi3)
    phase4, phaseBit4, pattern4 = point_2_phi_pattern(theta4, phi4)
    # GA-AP算法
    # 使用 相位叠加法(PS) 得到初始位置
    logger.info("[STEP 1] use PS to get init phase.")
    phase_mix_ps, phaseBit_mix_ps = psm_beam_4(phase1, phase2, phase3, phase4)
    # 使用 交替投影法 设置GA的fitness
    logger.info("[STEP 2] use GA-AP to get optimized phase.")
    ga = GA()
    phaseBit_mix, fitness, fitness_list = ga.beamScanning2Dual(theta1, phi1, theta2, phi2, phaseBit_mix_ps)
    logger.info("fitness: %f" % (fitness))
    # 计算phase_mix的方向图
    phaseBit_mix = np.array(phaseBit_mix)
    patternBit_mix = phase_2_pattern(phaseBit_mix)
    #
    # 保存结果
    logger.info("save GA-AP multi-beam 4 result...")
    path_pre = "../../files/multi-beam/GA-AP/4-(30,0,30,90,30,180,30,270)/"
    # 保存图片
    save_img(path_pre + "phase1.jpg", phase1)
    save_img(path_pre + "phase2.jpg", phase2)
    save_img(path_pre + "phase3.jpg", phase3)
    save_img(path_pre + "phase4.jpg", phase4)
    save_img(path_pre + "phaseBit1.jpg", phaseBit1)
    save_img(path_pre + "phaseBit2.jpg", phaseBit2)
    save_img(path_pre + "phaseBit3.jpg", phaseBit3)
    save_img(path_pre + "phaseBit4.jpg", phaseBit4)
    save_img(path_pre + "pattern1.jpg", pattern1)
    save_img(path_pre + "pattern2.jpg", pattern2)
    save_img(path_pre + "pattern3.jpg", pattern3)
    save_img(path_pre + "pattern4.jpg", pattern4)
    save_img(path_pre + "phaseBit_mix.jpg", phaseBit_mix)         # GA法 -- 结果码阵
    save_img(path_pre + "patternBit_mix.jpg", patternBit_mix)     # GA法 -- 结果码阵方向图
    # 保存相位结果
    save_csv(phase1, path_pre + "phase1.csv")
    save_csv(phase2, path_pre + "phase2.csv")
    save_csv(phase3, path_pre + "phase3.csv")
    save_csv(phase4, path_pre + "phase4.csv")
    save_csv(phaseBit1, path_pre + "phaseBit1.csv")
    save_csv(phaseBit2, path_pre + "phaseBit2.csv")
    save_csv(phaseBit3, path_pre + "phaseBit3.csv")
    save_csv(phaseBit4, path_pre + "phaseBit4.csv")
    save_csv(phaseBit_mix, path_pre + "phaseBit_mix.csv")
    #fitness_list = [float(x) for x in fitness_list]
    #save_csv(fitness_list, path_pre + "fitness.csv")


if __name__ == '__main__':
    # 配置日志，默认打印到控制台，也可以设置打印到文件
    setup_logging()
    # setup_logging(log_file="../../files/logs/log_multi_beam_PS.log")

    # 获取日志记录器并记录日志
    logger = logging.getLogger("[RIS-multi-beam-PS-1bit]")
    logger.info("1bit-RIS-multi-beam-PS: GA based Alternating projection method")
    # 基于GA(寻优方法) - 交替投影(GA的fitness) - 相位叠加(初始位置) 的方法: 主函数
    # testGA()
    # main_multi_beam_2(30, 0, 30, 180)
    main_multi_beam_4(30, 0, 30, 90, 30, 180, 30, 270)