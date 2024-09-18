import numpy as np
from sklearn.cluster import KMeans
from chebyshev.arr_chebyshev_plane import arr_chebyshev_plane
from chebyshev.arr_chebyshev_line_3 import arr_chebyshev_line

# 计算两个数据点之间的欧氏距离
def euclidean_distance(weights, x1, x2):
    # return np.sqrt(np.sum(weights * (x1 - x2) ** 2, axis=-1))
    return np.sqrt(np.sum(weights * (x1 - x2) ** 2))
    # return np.sqrt(np.sum((x1 - x2) ** 2))


# K-means聚类算法
def weighted_kmeans(X, k, weights, max_iters=100):
    # 随机选择k个中心点
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx]

    for _ in range(max_iters):
        # 分配每个数据点到最近的中心点
        clusters = [[] for _ in range(k)]
        for i, x in enumerate(X):
            distances = [euclidean_distance(weights, x, centroid) for centroid in centroids]
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


'''
# 创建加权KMeans模型
class WeightedKMeans(KMeans):
    def __init__(self, n_clusters=8, weights=None, **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.weights = weights

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum(self.weights * (a - b) ** 2, axis=-1))

def weighted_kmeans(X, k, weights, max_iters=100):
    kmeans = WeightedKMeans(n_clusters=k, weights=weights)
    # 使用数据进行训练
    kmeans.fit(X)
    # 获取每个样本的所属簇标签
    labels = kmeans.labels_
    # 获取聚类中心点的坐标
    centroids = kmeans.cluster_centers_
    # print("样本标签:", labels)
    # print("聚类中心点坐标:", centroids)
    return labels, centroids
'''

'''
def weighted_kmeans(X, k, weights, max_iters=100):
    # 初始化k个质心
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    for _ in range(max_iters):
        # 计算每个数据点到各个质心的加权距离
        distances = np.zeros((len(X), k))
        for i in range(k):
            distances[:, i] = np.sqrt(np.sum(weights * (X - centroids[i]) ** 2, axis=1))

        # 将每个数据点分配给加权距离最小的质心所属的簇
        labels = np.argmin(distances, axis=1)

        # 更新质心位置，考虑数据点的权重
        new_centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            cluster_points = X[labels == i]
            new_centroids[i] = np.average(cluster_points, axis=0, weights=weights[labels == i])

        # 如果质心不再发生明显变化，则停止迭代
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids
'''



def weighted_kmeans_plane(lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps, k_kmeans, weights_kmeans):
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
    #
    # 将权重数组扩展到与数据点相同的维度
    weights_kmeans = np.tile(weights_kmeans[:, np.newaxis], (1, X.shape[1]))
    clusters, centroids = weighted_kmeans(X=X, k=k_kmeans, weights=weights_kmeans, max_iters=10)
    return clusters, centroids


def weighted_kmeans_line(lambda_, N, SLL, d, theta0, NN, k_kmeans, weights_kmeans):
    # 直线阵 -- 切比雪夫方法
    weights, pattern_dbw, theta = arr_chebyshev_line(lambda_, N, SLL, d, theta0, NN)
    # 转换合适格式
    arr_weights = list()
    for weight in weights:
        arr_weights.append([weight])
    np_arr_weights = np.array(arr_weights)
    # print("np_arr_weights:", X111)
    X = np_arr_weights
    # print("X:", X)
    # 将权重数组扩展到与数据点相同的维度
    weights_kmeans = np.tile(weights_kmeans[:, np.newaxis], (1, X.shape[1]))
    clusters, centroids = weighted_kmeans(X=X, k=k_kmeans, weights=weights_kmeans, max_iters=10)
    return clusters, centroids




def test_weighted_kmeans():
    # 示例用法
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [9, 8]])
    k = 2
    weights = np.array([0.5, 0.7, 1.0, 0.3, 0.8, 1.0])

    # 将权重数组扩展到与数据点相同的维度
    weights = np.tile(weights[:, np.newaxis], (1, X.shape[1]))

    labels, centroids = weighted_kmeans(X, k, weights)
    print("簇分配结果：", labels)
    print("质心位置：", centroids)


def test_weighted_kmeans_plane():
    # 初始化参数
    lambda_ = 1  # 波长
    d = 0.5  # 阵元间隔
    Ny = 15  # 方位阵元个数
    Nz = 15  # 俯仰阵元个数
    phi0 = 0  # 方位指向
    theta0 = 0  # 俯仰指向
    eps = 0.0001  # 底电平
    NA = 360  # 方位角度采样
    NE = 360  # 俯仰角度采样
    SLL = -30
    k_kmeans = 10  # 子阵数量
    #
    # weights_kmeans = np.array([0.5, 0.7, 1.0, 0.3, 0.8, 1.0])
    weights_kmeans = np.random.rand(Ny*Nz)

    labels, centroids = weighted_kmeans_plane(lambda_=lambda_, Ny=Ny, Nz=Nz, SLL=SLL, d=d,
                                              phi0=phi0, theta0=theta0, NA=NA, NE=NE, eps=eps,
                                              k_kmeans=k_kmeans, weights_kmeans=weights_kmeans)
    print("簇分配结果：", labels)
    print("质心位置：", centroids)



if __name__ == '__main__':
    test_weighted_kmeans()
    # test_weighted_kmeans_plane()