import numpy as np


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


if __name__ == '__main__':
    test_weighted_kmeans()