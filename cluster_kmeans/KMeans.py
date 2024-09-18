import numpy as np

class KMeans():
    clusters = None
    centroids = None

    # 计算两个数据点之间的欧氏距离
    def __euclidean_distance(self, x1, x2):
        # distance = 0
        # for i in range(len(x1)):
        #     distance += abs(x1[i] - x2[i])
        # return distance
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # 计算两个数据点之间的加权欧氏距离
    def __euclidean_distance_weights(self, weights, x1, x2):
        # return np.sqrt(np.sum(weights * (x1 - x2) ** 2, axis=-1))
        return np.sqrt(np.sum(weights * (x1 - x2) ** 2))

    # K-means 聚类算法
    def parseKMeans(self, X, k, max_iters=100, initial_centroids=None):
        # 如果提供了初始中心点，则使用它们，否则随机选择k个中心点
        if initial_centroids is not None:
            print("kmeans plane with init:", initial_centroids)
            # assert initial_centroids.shape == (k, X.shape[1]), "initial_centroids shape mismatch"
            self.centroids = X[initial_centroids]
        else:
            idx = np.random.choice(len(X), k, replace=False)
            self.centroids = X[idx]

        for _ in range(max_iters):
            # 分配每个数据点到最近的中心点
            self.clusters = [[] for _ in range(k)]
            for i, x in enumerate(X):
                distances = [self.__euclidean_distance(x, centroid) for centroid in self.centroids]
                cluster_idx = np.argmin(distances)
                self.clusters[cluster_idx].append(i)

            # 更新中心点
            prev_centroids = self.centroids.copy()
            for i, cluster in enumerate(self.clusters):
                if len(cluster) > 0:
                    cluster_points = X[cluster]
                    self.centroids[i] = cluster_points.mean(axis=0)

            # 判断是否收敛
            if np.all(prev_centroids == self.centroids):
                break

        return self.clusters, self.centroids

    # 加权 K-means 聚类算法
    def parseKMeansWeighted(self, weights, X, k, max_iters=100, initial_centroids=None):
        # 如果提供了初始中心点，则使用它们，否则随机选择k个中心点
        if initial_centroids is not None:
            print("kmeans plane with init:", initial_centroids)
            # assert initial_centroids.shape == (k, X.shape[1]), "initial_centroids shape mismatch"
            self.centroids = X[initial_centroids]
        else:
            idx = np.random.choice(len(X), k, replace=False)
            self.centroids = X[idx]

        for _ in range(max_iters):
            # 分配每个数据点到最近的中心点
            self.clusters = [[] for _ in range(k)]
            for i, x in enumerate(X):
                distances = [self.__euclidean_distance_weights(weights, x, centroid) for centroid in self.centroids]
                cluster_idx = np.argmin(distances)
                self.clusters[cluster_idx].append(i)

            # 更新中心点
            prev_centroids = self.centroids.copy()
            for i, cluster in enumerate(self.clusters):
                if len(cluster) > 0:
                    cluster_points = X[cluster]
                    self.centroids[i] = cluster_points.mean(axis=0)

            # 判断是否收敛
            if np.all(prev_centroids == self.centroids):
                break

        return self.clusters, self.centroids