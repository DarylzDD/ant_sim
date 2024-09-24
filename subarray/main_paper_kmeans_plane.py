import csv
import statistics
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import chebwin

from cluster_kmeans.kmeans_plane import kmeans_plane
from cluster_kmeans.weighted_kmeans_plane import weighted_kmeans_plane

from cluster_kmeans.cluster_kmeans_jaccard import calc_jaccard_mean
from read_fBest_mat_plane import get_spare_weight
from chebyshev.arr_chebyshev_plane import compute_array_pattern

from util.util_analysis_plane import get_peaks

from subarray.SubarrayClusterKMeans import SubarrayClusterKMeans


def generate_arr_w(Ny, Nz, clusters, centroids):
    arr_w = np.zeros((Ny, Nz))
    for i, cluster in enumerate(clusters):
        # print(f"Cluster {i + 1}: {cluster} - {centroids[i]}")
        for item in cluster:
            if len(cluster) == 0:
                continue
            x = item // Nz
            y = item % Nz
            # print("item=%d, x=%d, y=%d" % (item, x, y))
            arr_w[x][y] = centroids[i]
    # print("arr_w:", arr_w)
    return arr_w


def psl_plane(lambda_, Ny, Nz, NA, NE, d, arr_w, theta0, phi0, eps):
    # 1.计算方向图
    pattern_dbw, theta, phi = compute_array_pattern(lambda_, d, Ny, Nz, theta0, phi0, arr_w, NA, NE, eps)
    # 2.找峰
    peaks = get_peaks(pattern_dbw)
    # 3.找最大副瓣
    psl = None
    psl_val = 1
    for i in range(len(peaks)):
        peak = peaks[i]
        peak_val = peak[0]
        if peak_val < -3 and i > 0:
            psl = peak
            psl_val = peak_val
            break
    return psl_val


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
        writer.writerows(data)
    print("数据已成功保存到CSV文件:", file_path)


def cluster_loop_kmeans_wrapper(mode, lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps, k_kmeans, weights_kmeans, initial_centroids):
    if mode == 1:
        clusters, centroids = kmeans_plane(lambda_=lambda_, Ny=Ny, Nz=Nz, SLL=SLL, d=d, phi0=phi0, theta0=theta0, NA=NA,
                                           NE=NE, eps=eps, k_kmeans=k_kmeans, initial_centroids=initial_centroids)
    elif mode == 2:
        clusters, centroids = weighted_kmeans_plane(lambda_=lambda_, Ny=Ny, Nz=Nz, SLL=SLL, d=d, phi0=phi0, theta0=theta0, NA=NA,
                                                    NE=NE, eps=eps, k_kmeans=k_kmeans, weights_kmeans=weights_kmeans)
    elif mode == 3:
        clusters, centroids = kmeans_plane(lambda_=lambda_, Ny=Ny, Nz=Nz, SLL=SLL, d=d, phi0=phi0, theta0=theta0, NA=NA,
                                           NE=NE, eps=eps, k_kmeans=k_kmeans, initial_centroids=initial_centroids)
    elif mode == 4:
        # initial_centroids = np.array([14, 23, 31, 51, 84, 80, 108, 116, 142, 153])  # 初始中心点，来自稀疏结果
        # initial_centroids = np.array([14, 23, 31, 50, 52, 78, 95, 84, 98, 116, 108, 126, 155, 145, 153])
        # sorted_indices_centroids = np.array(
        #     [142, 36, 54, 80, 81, 96, 109, 112, 125, 144, 23, 31, 49, 82, 101, 117, 123, 127, 153, 154, 170, 24, 63, 67,
        #      85, 86, 93, 95, 97, 99, 103, 110, 111, 115, 128, 132, 141, 143, 146, 156, 158, 161, 169, 171, 15, 25, 33,
        #      52, 68, 69])
        #
        # initial_centroids_candidate = [142, 36, 54, 80, 81, 96, 109, 112, 125, 144, 23, 31, 49, 82, 101, 117, 123, 127,
        #                                153, 154, 170, 24, 63, 67, 85, 86, 93, 95, 97, 99, 103, 110, 111, 115, 128, 132,
        #                                141, 143, 146, 156, 158, 161, 169, 171, 15, 25, 33, 52, 68, 69]
        # initial_centroids_new = np.array(random.sample(initial_centroids_candidate, 15))
        sck = SubarrayClusterKMeans()
        # print("initial_centroids_new:", initial_centroids_new)
        # clusters, centroids = sck.planeWeightedInitial(lambda_=lambda_, Ny=Ny, Nz=Nz, SLL=SLL, d=d, phi0=phi0, theta0=theta0, NA=NA,
        #                                    NE=NE, eps=eps, k_kmeans=k_kmeans, weights_kmeans=weights_kmeans, initial_centroids=initial_centroids_new)
        #
        clusters, centroids = sck.planeWeightedPSO(lambda_=lambda_, Ny=Ny, Nz=Nz, SLL=SLL, d=d, phi0=phi0, theta0=theta0,
                                                   NA=NA, NE=NE, eps=eps, k_kmeans=k_kmeans, weights_kmeans=weights_kmeans)
    else:
        clusters, centroids = kmeans_plane(lambda_=lambda_, Ny=Ny, Nz=Nz, SLL=SLL, d=d, phi0=phi0, theta0=theta0, NA=NA,
                                           NE=NE, eps=eps, k_kmeans=k_kmeans, initial_centroids=None)
    return clusters, centroids


def cluster_loop(mode, lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps, k_kmeans, weights_kmeans, initial_centroids):
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
    dict_mode = {1: "kmeans", 2: "kmeans_weighted", 3: "kmeans_init", 4: "kmeans_init_weighted"}
    #
    print("MODE: %d, %s" % (mode, dict_mode[mode]))
    for i in range(0, 30):
        print("i = %i" % (i))
        cc = dict()
        #
        clusters, centroids = cluster_loop_kmeans_wrapper(mode, lambda_, Ny, Nz, SLL, d, math.radians(phi0), math.radians(theta0), NA, NE, eps, k_kmeans, weights_kmeans, initial_centroids)
        #
        arr_w = generate_arr_w(Ny=Ny, Nz=Nz, clusters=clusters, centroids=centroids)
        arr_w_psl = psl_plane(lambda_=lambda_, Ny=Ny, Nz=Nz, NA=NA, NE=NE, d=d, arr_w=arr_w, theta0=theta0, phi0=phi0, eps=eps)
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
            arr1 = list(itertools.chain(*arr_w))
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
        arr1 = list(itertools.chain(*arr_w))
        # print(arr1)
        arr2 = [int(x * 1000000) for x in arr1]
        list_arr_w_line.append(arr2)
    save_csv(list_arr_w_line, path_file)


def get_weight_6(Ny, Nz, SLL):
    Ty = chebwin(Ny, abs(SLL))
    Tz = chebwin(Nz, abs(SLL))
    # 生成权重矩阵
    weights = np.outer(Ty, Tz)
    return weights.flatten()


def main_reference_6():
    # 【0. 初始化参数】
    lambda_ = 1  # 波长
    d = 0.5  # 阵元间隔
    Ny = 40  # 方位阵元个数
    Nz = 40  # 俯仰阵元个数
    phi0 = 0  # 方位指向
    theta0 = 0  # 俯仰指向
    eps = 0.0001  # 底电平
    NA = 360  # 方位角度采样
    NE = 360  # 俯仰角度采样
    SLL = -60
    k_kmeans = 9  # 子阵数量
    test_idx = "005"
    #
    # 【1. 稀疏结果处理】
    # weights_kmeans = np.full(Ny*Nz, 0.566)
    # weights_kmeans = np.array(get_spare_weight())   # 稀疏结果做加权值
    weights_kmeans = get_weight_6(Ny, Nz, SLL)   # 参考文献[6]做加权值
    # initial_centroids = np.array([14, 23, 31, 51, 84, 80, 108, 116, 142, 153])  # 初始中心点，来自稀疏结果
    #
    # 【2. 获取各种聚类结果】
    print("---------------------------begin cluster-----------------------------")
    #
    list_arr_w_kmeans_weighted, list_psl_kmeans_weighted, \
    best_clusters_kmeans_weighted, best_centroids_kmeans_weighted, \
    best_arr_kmeans_weighted, worst_arr_kmeans_weighted = cluster_loop(2, lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE,
                                                                       eps, k_kmeans, weights_kmeans, None)
    print("--------------------------- end cluster-----------------------------")
    #
    print("---------------------------begin result-----------------------------")
    #
    print("-------------Jaccard--------------")
    result_jaccard(list_arr_w_kmeans_weighted, "weighted-K-means reference [6]")
    #
    print("-------------PSL--------------")
    psl_max_kmeans_weighted, psl_min_kmeans_weighted, psl_mean_kmeans_weighted, psl_std_kmeans_weighted = calculate_statistics(
        list_psl_kmeans_weighted)
    print("[weighted-K-means] psl_max=%.6f, psl_min=%.6f, psl_mean=%.6f, psl_std=%.6f" % (
    psl_max_kmeans_weighted, psl_min_kmeans_weighted, psl_mean_kmeans_weighted, psl_std_kmeans_weighted))
    print("---------------------------end result-----------------------------")
    # 保存arr_w到csv
    path_arr_w_pre = "./files/subarray/" + str(Ny) + "x" + str(Nz) + "-" + str(k_kmeans) + "/"
    result_save_csv(list_arr_w=list_arr_w_kmeans_weighted,
                    path_file=path_arr_w_pre + "arr_w_[6]_wkcm_" + test_idx + ".csv")
    #
    save_csv(best_arr_kmeans_weighted, path_arr_w_pre + "best_arr_w_[6]_wkcm_" + test_idx + ".csv")
    save_csv(worst_arr_kmeans_weighted, path_arr_w_pre + "worst_arr_w_[6]_wkcm_" + test_idx + ".csv")


def main_point(theta0, phi0, path_pre):
    # 【0. 初始化参数】
    lambda_ = 1  # 波长
    d = 0.5  # 阵元间隔
    Ny = 40  # 方位阵元个数
    Nz = 40  # 俯仰阵元个数
    # phi0 = 0  # 方位指向
    # theta0 = 0  # 俯仰指向
    eps = 0.0001  # 底电平
    NA = 360  # 方位角度采样
    NE = 360  # 俯仰角度采样
    SLL = -60
    k_kmeans = 9  # 子阵数量
    test_idx = "_(" + str(theta0) + "," + str(phi0) + ")-2024-08-27-001"
    print("theta0: %f, phi0: %f" % (theta0, phi0))
    #
    # 【1. 稀疏结果处理】
    weights_swkcm = np.array(get_spare_weight())   # 稀疏结果做加权值
    weights_wkcm = get_weight_6(Ny, Nz, SLL)        # 参考文献[6]加权
    #
    # 【2. 获取各种聚类结果】
    print("---------------------------begin cluster-----------------------------")
    list_arr_w_kcm, list_psl_kcm, \
    best_clusters_kcm, best_centroids_kcm, \
    best_arr_kcm, worst_arr_kcm = cluster_loop(1, lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps, k_kmeans,
                                               None, None)
    #
    list_arr_w_wkcm, list_psl_wkcm, \
    best_clusters_wkcm, best_centroids_wkcm, \
    best_arr_wkcm, worst_arr_wkcm = cluster_loop(2, lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps, k_kmeans,
                                                 weights_wkcm, None)
    #
    list_arr_w_swkcm, list_psl_swkcm, \
    best_clusters_swkcm, best_centroids_swkcm, \
    best_arr_swkcm, worst_arr_swkcm = cluster_loop(2, lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps, k_kmeans,
                                                   weights_swkcm, None)
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
    path_arr_w_pre = path_pre + str(Ny) + "x" + str(Nz) + "-" + str(k_kmeans) + "/"
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
    # 主函数
    # 参考文献[6]的方法
    # main_reference_6()
    # 波束指向改角度
    # main_point(0, 0)
    main_point(30, 0, "../files/subarray/")
    # main_point(45, 0)
    # main_point(0, 30)
    # main_point(60, 60)