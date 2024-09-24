import csv
import statistics
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import chebwin

from cluster_kmeans.kmeans_line import kmeans_line
from cluster_kmeans.weighted_kmeans_plane import weighted_kmeans_line

from util.util_analysis_line import msll_line
from cluster_kmeans.cluster_kmeans_jaccard import calc_jaccard_mean
from read_fBest_mat_line import get_spare_weight


def generate_arr_w(N, clusters, centroids):
    arr_w = np.zeros(N)
    for i, cluster in enumerate(clusters):
        # print(f"Cluster {i + 1}: {cluster} - {centroids[i]}")
        for item in cluster:
            if len(cluster) == 0:
                continue
            # print("item=%d, x=%d, y=%d" % (item, x, y))
            arr_w[item] = centroids[i]
    # print("arr_w:", arr_w)
    return arr_w


def psl_line(lambda_, N, NN, d, arr_w, theta0):
    # 计算波数
    k = 2 * np.pi / lambda_
    # 转换波束指向角到弧度
    steering_angle = np.radians(theta0)
    # 角度范围从-90度到90度（以弧度为单位）
    theta = np.linspace(-np.pi / 2, np.pi / 2, NN)
    # 初始化阵因子Array Factor(AF)
    AF = np.zeros(theta.shape, dtype=complex)
    # 计算阵因子
    for i, ang in enumerate(theta):
        elementPos = np.arange(N)  # 阵元位置矢量
        phaseDiff = k * d * (np.cos(np.pi / 2 - ang) - np.cos(np.pi / 2 - steering_angle))
        AF[i] = np.sum(arr_w * np.exp(1j * phaseDiff * elementPos))
    # 归一化阵因子
    # AF = np.abs(AF) / np.max(np.abs(AF))
    # AF转dB
    eps = 0.0001
    pattern_dbw = 20 * np.log10(np.abs(AF) / np.max(np.abs(AF)) + eps)
    #
    msll_arr_w = msll_line(pattern_dbw)
    return msll_arr_w


def calculate_statistics(lst):
    maximum = max(lst)
    minimum = min(lst)
    average = sum(lst) / len(lst)
    standard_deviation = statistics.stdev(lst)
    return maximum, minimum, average, standard_deviation


def save_csv(data, file_path):
    # 指定CSV文件路径
    # file_path = 'data.csv'
    # 保存数据到CSV文件
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerows(data)
        writer.writerow(data)
    print("数据已成功保存到CSV文件:", file_path)


def cluster_loop_kmeans_wrapper(mode, lambda_, N, SLL, d, theta0, NN, k_kmeans, weights_kmeans):
    if mode == 1:
        clusters, centroids = kmeans_line(lambda_=lambda_, N=N, SLL=SLL, d=d, k=k_kmeans, theta0=theta0, NN=NN)
    elif mode == 2:
        clusters, centroids = weighted_kmeans_line(lambda_=lambda_, N=N, SLL=SLL, d=d, theta0=theta0, NN=NN,
                                                   k_kmeans=k_kmeans, weights_kmeans=weights_kmeans)
    else:
        clusters, centroids = kmeans_line(lambda_=lambda_, N=N, SLL=SLL, d=d, k=k_kmeans, theta0=theta0, NN=NN)
    return clusters, centroids


def cluster_loop(mode, lambda_, N, SLL, d, theta0, NN, k_kmeans, weights_kmeans):
    list_cc = list()
    best_clusters = None
    best_centroids = None
    best_psl = 0
    best_arr = list()
    worst_psl = -60
    worst_arr = list()
    list_psl = list()
    list_clusters = list()
    list_centroids = list()
    list_arr_w = list()
    dict_mode = {1: "kmeans", 2: "kmeans_weighted"}
    #
    print("MODE: %d, %s" % (mode, dict_mode[mode]))
    for i in range(0, 3):
        print("i = %i" % (i))
        cc = dict()
        #
        clusters, centroids = cluster_loop_kmeans_wrapper(mode, lambda_, N, SLL, d, theta0, NN, k_kmeans, weights_kmeans)
        #
        arr_w = generate_arr_w(N=N, clusters=clusters, centroids=centroids)
        arr_w_psl = psl_line(lambda_=lambda_, N=N, NN=NN, d=d, arr_w=arr_w, theta0=theta0)
        cc['clusters'] = clusters
        cc['centroids'] = centroids
        cc['arr_w'] = arr_w
        cc['psl'] = arr_w_psl
        list_psl.append(arr_w_psl)
        list_clusters.append(clusters)
        list_centroids.append(centroids)
        list_arr_w.append(arr_w.tolist())
        list_cc.append(cc)
        if arr_w_psl < best_psl:
            print("get better psl. best=%.6f, new=%.6f" % (best_psl, arr_w_psl))
            best_psl = arr_w_psl
            best_arr = arr_w
            best_clusters = clusters
            best_centroids = centroids
        if arr_w_psl > worst_psl:
            print("get worse psl. worst=%.6f, new=%.6f" % (worst_psl, arr_w_psl))
            worst_psl = arr_w_psl
            worst_arr = arr_w
    #
    return list_arr_w, list_psl, best_clusters, best_centroids, best_arr, worst_arr

def result_jaccard(list_arr_w, mode_name):
    list_arr_w_line = []
    if len(list_arr_w) > 2:
        for arr_w in list_arr_w:
            # print(arr_w)
            # arr1 = list(itertools.chain(*arr_w))
            arr1 = arr_w
            # print(arr1)
            arr2 = [int(x * 100) for x in arr1]
            list_arr_w_line.append(arr2)
            # print(arr2)
        jaccard = calc_jaccard_mean(list_arr_w_line)
        print("[%s] - jaccard = %.6f" % (mode_name, jaccard))
    else:
        print("length of list_arr_w less than 2. no jaccard")



def result_save_csv(list_arr_w, path_file):
    list_arr_w_line = []
    for arr_w in list_arr_w:
        # print(arr_w)
        # arr1 = list(itertools.chain(*arr_w))
        arr1 = arr_w
        # print(arr1)
        arr2 = [int(x * 1000000) for x in arr1]
        list_arr_w_line.append(arr2)
    save_csv(list_arr_w_line, path_file)


def get_weight_6(N, SLL):
    a = chebwin(N, abs(SLL))
    return a


def main_point(theta0, path_pre):
    # 【0. 初始化参数】
    lambda_ = 1  # 波长
    d = 0.5  # 阵元间隔
    N = 100  # 阵元个数
    eps = 0.0001  # 底电平
    NN = 1000  # 方位角度采样
    SLL = -60
    k_kmeans = 9  # 子阵数量
    test_idx = "_(" + str(theta0) + ")-2024-09-18-001"
    print("theta0: %f" % (theta0))
    #
    # 【1. 稀疏结果处理】
    weights_swkcm = np.array(get_spare_weight())   # 稀疏结果做加权值
    weights_wkcm = get_weight_6(N, SLL)        # 参考文献[6]加权
    #
    # 【2. 获取各种聚类结果】
    print("---------------------------begin cluster-----------------------------")
    list_arr_w_kcm, list_psl_kcm, \
    best_clusters_kcm, best_centroids_kcm, \
    best_arr_kcm, worst_arr_kcm = cluster_loop(1, lambda_, N, SLL, d, theta0, NN, k_kmeans, None)
    #
    list_arr_w_wkcm, list_psl_wkcm, \
    best_clusters_wkcm, best_centroids_wkcm, \
    best_arr_wkcm, worst_arr_wkcm = cluster_loop(2, lambda_, N, SLL, d, theta0, NN, k_kmeans, weights_wkcm)
    #
    list_arr_w_swkcm, list_psl_swkcm, \
    best_clusters_swkcm, best_centroids_swkcm, \
    best_arr_swkcm, worst_arr_swkcm = cluster_loop(2, lambda_, N, SLL, d, theta0, NN, k_kmeans, weights_swkcm)
    #
    print("--------------------------- end cluster-----------------------------")
    print("---------------------------begin result-----------------------------")
    # 算相似度
    print("-------------Jaccard--------------")
    result_jaccard(list_arr_w_kcm, "KCM")
    result_jaccard(list_arr_w_wkcm, "WKCM")
    result_jaccard(list_arr_w_swkcm, "SWKCM")
    # 算特征
    print("-------------PSL--------------")
    psl_max_kcm, psl_min_kcm, psl_mean_kcm, psl_std_kcm = calculate_statistics(list_psl_kcm)
    psl_max_wkcm, psl_min_wkcm, psl_mean_wkcm, psl_std_wkcm = calculate_statistics(list_psl_wkcm)
    psl_max_swkcm, psl_min_swkcm, psl_mean_swkcm, psl_std_swkcm = calculate_statistics(list_psl_swkcm)
    print("[KCM] psl_max=%.6f, psl_min=%.6f, psl_mean=%.6f, psl_std=%.6f"
          % (psl_max_kcm, psl_min_kcm, psl_mean_kcm, psl_std_kcm))
    print("[WKCM] psl_max=%.6f, psl_min=%.6f, psl_mean=%.6f, psl_std=%.6f"
          % (psl_max_wkcm, psl_min_wkcm, psl_mean_wkcm, psl_std_wkcm))
    print("[SWKCM] psl_max=%.6f, psl_min=%.6f, psl_mean=%.6f, psl_std=%.6f"
          % (psl_max_swkcm, psl_min_swkcm, psl_mean_swkcm, psl_std_swkcm))
    print("---------------------------end result-----------------------------")
    # 保存arr_w到csv
    path_arr_w_pre = path_pre + str(N) + "-" + str(k_kmeans) + "/"
    result_save_csv(list_arr_w=list_arr_w_kcm, path_file=path_arr_w_pre + "arr_w_kcm_" + test_idx + ".csv")
    result_save_csv(list_arr_w=list_arr_w_wkcm, path_file=path_arr_w_pre + "arr_w_wkcm_" + test_idx + ".csv")
    result_save_csv(list_arr_w=list_arr_w_swkcm, path_file=path_arr_w_pre + "arr_w_swkcm_" + test_idx + ".csv")
    #
    save_csv(best_arr_kcm, path_arr_w_pre + "best_arr_w_kcm_" + test_idx + ".csv")
    save_csv(best_arr_wkcm, path_arr_w_pre + "best_arr_w_wkcm_" + test_idx + ".csv")
    save_csv(best_arr_swkcm, path_arr_w_pre + "best_arr_w_swkcm_" + test_idx + ".csv")
    save_csv(worst_arr_kcm, path_arr_w_pre + "worst_arr_w_kcm_" + test_idx + ".csv")
    save_csv(worst_arr_wkcm, path_arr_w_pre + "worst_arr_w_wkcm_" + test_idx + ".csv")
    save_csv(worst_arr_swkcm, path_arr_w_pre + "worst_arr_w_swkcm_" + test_idx + ".csv")


if __name__ == '__main__':
    # 主函数: 波束指向改角度
    # main_point(0)
    main_point(15, "../files/subarray/res/")
    # main_point(30)
