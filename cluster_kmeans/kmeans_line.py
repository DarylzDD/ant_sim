import numpy as np
from chebyshev.arr_chebyshev_line_3 import arr_chebyshev_line


# 计算两个数据点之间的欧氏距离
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# K-means聚类算法
def kmeans(X, k, max_iters=100):
    # 随机选择k个中心点
    idx = np.random.choice(len(X), k, replace=False)
    # list_idx = [5, 15, 29, 41, 51, 61, 71, 85, 95]
    # idx = np.array(list_idx)
    centroids = X[idx]

    for _ in range(max_iters):
        # 分配每个数据点到最近的中心点
        clusters = [[] for _ in range(k)]
        for i, x in enumerate(X):
            distances = [euclidean_distance(x, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(i)

        # 更新中心点
        prev_centroids = centroids.copy()
        for i, cluster in enumerate(clusters):
            if len(cluster) > 0:
                cluster_points = X[cluster]
                centroids[i] = cluster_points.mean(axis=0)

        # 判断是否收敛
        if np.all(prev_centroids == centroids):
            break

    return clusters, centroids


def test_kmeans():
    # 测试算法
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    k = 2
    clusters, centroids = kmeans(X, k)
    print("聚类结果：")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")
    print("中心点：")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i + 1}: {centroid}")


def kmeans_line(lambda_, N, SLL, d, k, theta0, NN):
    # 测试算法
    X1, AF, theta_deg = arr_chebyshev_line(lambda_, N, SLL, d, theta0, NN)
    # print("X1:", X1)
    X11 = list()
    for x in X1:
        X11.append([x])
    # print("X11:", X11)
    X111 = np.array(X11)
    # print("X111:", X111)
    #
    X = X111
    # X = np.array([[1], [1.5], [1.6], [4], [4.1], [4.2]])
    # print("X:", X)
    # k = 9
    clusters, centroids = kmeans(X, k)
    # print("聚类结果：")
    # for i, cluster in enumerate(clusters):
    #     print(f"Cluster {i + 1}: {cluster}")
    # print("中心点：")
    # for i, centroid in enumerate(centroids):
    #     print(f"Centroid {i + 1}: {centroid}")
    return clusters, centroids


if __name__ == '__main__':
    # test_kmeans()
    #
    #
    # 阵列参数
    N = 100  # 阵元数
    d = 0.5  # 阵元间距（以波长为单位）
    lambda_ = 1  # 波长
    SLL = -30  # 副瓣电平，以分贝为单位
    k = 9  # 子阵数量
    # 1.聚类
    clusters, centroids = kmeans_line(lambda_=lambda_, N=N, SLL=SLL, d=d, k=k)
    print("聚类结果：")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")
    print("中心点：")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i + 1}: {centroid}")