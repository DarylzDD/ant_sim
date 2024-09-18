import numpy as np


# 计算两个数据点之间的欧氏距离
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# K-means聚类算法
def kmeans(X, k, max_iters=100):
    # 随机选择k个中心点
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
