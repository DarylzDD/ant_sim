import numpy as np
from chebyshev.arr_chebyshev_plane import arr_chebyshev_plane


# 计算两个数据点之间的欧氏距离
def euclidean_distance(x1, x2):
    # distance = 0
    # for i in range(len(x1)):
    #     distance += abs(x1[i] - x2[i])
    # return distance
    return np.sqrt(np.sum((x1 - x2) ** 2))




# K-means聚类算法
def kmeans(X, k, max_iters=100, initial_centroids=None):
    # 如果提供了初始中心点，则使用它们，否则随机选择k个中心点
    if initial_centroids is not None:
        print("kmeans plane with init:", initial_centroids)
        # assert initial_centroids.shape == (k, X.shape[1]), "initial_centroids shape mismatch"
        centroids = X[initial_centroids]
    else:
        idx = np.random.choice(len(X), k, replace=False)
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

def test_kmeans2():
    # 假设 X 是你的数据集，它是一个 NumPy 数组
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    k = 2
    initial_centroids = np.array([[1, 2], [3, 4]])  # 举例的初始中心点，应与你数据相匹配
    clusters, centroids = kmeans(X=X, k=k, max_iters=100, initial_centroids=initial_centroids)
    print("聚类结果：")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")
    print("中心点：")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i + 1}: {centroid}")


def test_kmeans3():
    # 假设 X 是你的数据集，它是一个 NumPy 数组
    X = np.array([1, 12, 100, 111, 13, 61, 0]).reshape(-1, 1)
    k = 2
    initial_centroids = np.array([1, 40]).reshape(k, 1)  # 举例的初始中心点，应与你数据相匹配
    clusters, centroids = kmeans(X=X, k=k, max_iters=100, initial_centroids=initial_centroids)
    print("聚类结果：")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")
    print("中心点：")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i + 1}: {centroid}")


def kmeans_plane(lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps, k_kmeans, initial_centroids=None):
    # 测试算法
    weights, pattern_dbw, theta, phi = arr_chebyshev_plane(lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps)
    # print("X1:", X1)
    X11 = list()
    for xs in weights:
        for x in xs:
            X11.append([x])
    # print("X11:", X11)
    X111 = np.array(X11)
    # print("X111:", X111)
    #
    X = X111
    # print("X:", X)
    clusters, centroids = kmeans(X, k_kmeans, max_iters=10, initial_centroids=initial_centroids)
    return clusters, centroids


if __name__ == '__main__':
    # test_kmeans()
    test_kmeans3()
    # # 初始化参数
    # lambda_ = 1  # 波长
    # d = 0.5  # 阵元间隔
    # Ny = 20  # 方位阵元个数
    # Nz = 10  # 俯仰阵元个数
    # phi0 = 0  # 方位指向
    # theta0 = 0  # 俯仰指向
    # eps = 0.0001  # 底电平
    # NA = 360  # 方位角度采样
    # NE = 360  # 俯仰角度采样
    # SLL = -30
    # # 1.聚类
    # clusters, centroids = kmeans_plane(lambda_=lambda_, Ny=Ny, Nz=Nz, SLL=SLL, d=d, phi0=phi0, theta0=theta0, NA=NA, NE=NE, eps=eps, k_kmeans=20)
    # print("聚类结果：")
    # for i, cluster in enumerate(clusters):
    #     print(f"Cluster {i + 1}: {cluster}")
    # print("中心点：")
    # for i, centroid in enumerate(centroids):
    #     print(f"Centroid {i + 1}: {centroid}")