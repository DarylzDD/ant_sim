import numpy as np

from chebyshev.arr_chebyshev_plane import arr_chebyshev_plane

from cluster_kmeans.KMeans import KMeans
from cluster_kmeans.KMeansPSO import KMeansPSO


class SubarrayClusterKMeans():

    def planeWeightedInitial(self, lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps, k_kmeans, weights_kmeans, initial_centroids):
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
        k_means = KMeans()
        clusters, centroids = k_means.parseKMeansWeighted(weights=weights_kmeans, X=X, k=k_kmeans, max_iters=100, initial_centroids=initial_centroids)
        return clusters, centroids


    def planeWeightedPSO(self, lambda_, Ny, Nz, SLL, d, phi0, theta0, NA, NE, eps, k_kmeans, weights_kmeans):
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
        kp = KMeansPSO()
        print("kp.kmeans_pso_weighted!!!!!!!!!")
        clusters, centroids = kp.kmeans_pso_weighted(X=X, weights=weights_kmeans, n_clusters=k_kmeans, n_particles=30, n_iterations=100)
        return clusters, centroids