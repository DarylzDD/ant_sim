import numpy as np
from sklearn.datasets import make_blobs


class KMeansPSO():
    def __distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def __distance_weights(self, weights, x1, x2):
        return np.sqrt(np.sum(weights * (x1 - x2) ** 2))

    def kmeans_pso(self, X, n_clusters, n_particles, n_iterations):
        n_samples, n_features = X.shape

        # 初始化粒子群
        particles = np.random.uniform(low=np.min(X), high=np.max(X), size=(n_particles, n_clusters, n_features))

        # 初始化粒子最佳位置和全局最佳位置
        particle_best_positions = particles.copy()
        global_best_position = particles[
            np.argmin([np.sum([self.__distance(X[i], particle) for i in range(n_samples)]) for particle in particles])]

        # 初始化粒子速度
        velocities = np.zeros_like(particles)

        for _ in range(n_iterations):
            for i in range(n_particles):
                for j in range(n_clusters):
                    # 计算粒子位置对应的适应度值
                    inertia = np.sum([self.__distance(X[k], particles[i, j]) for k in range(n_samples)])
                    particle_best_inertia = np.sum(
                        [self.__distance(X[k], particle) for k in range(n_samples) for particle in particle_best_positions[i]])
                    global_best_inertia = np.sum(
                        [self.__distance(X[k], particle) for k in range(n_samples) for particle in global_best_position])

                    # 更新粒子速度和位置
                    velocities[i, j] = velocities[i, j] + np.random.rand() * (
                                particle_best_positions[i, j] - particles[i, j]) + np.random.rand() * (
                                                   global_best_position[j] - particles[i, j])
                    particles[i, j] = particles[i, j] + velocities[i, j]

                    # 限制粒子位置在数据范围内
                    particles[i, j] = np.clip(particles[i, j], np.min(X), np.max(X))

            # 更新粒子最佳位置和全局最佳位置
            for i in range(n_particles):
                if np.sum([self.__distance(X[k], particles[i, j]) for k in range(n_samples) for j in
                           range(n_clusters)]) < np.sum(
                        [self.__distance(X[k], particle) for k in range(n_samples) for particle in
                         particle_best_positions[i]]):
                    particle_best_positions[i] = particles[i].copy()

            if np.sum(
                    [self.__distance(X[k], particle) for k in range(n_samples) for particle in global_best_position]) > np.sum(
                    [self.__distance(X[k], particle) for k in range(n_samples) for particle in particle_best_positions[i]]):
                global_best_position = particle_best_positions[i].copy()

        # 根据最终的全局最佳位置进行聚类
        clusters = []
        centroids = []
        for sample in X:
            distances = [self.__distance(sample, centroid) for centroid in global_best_position]
            label = np.argmin(distances)
            clusters.append(label)
        for centroid in global_best_position:
            centroids.append(centroid)

        # return np.array(clusters), np.array(centroids)
        return clusters, centroids

    def kmeans_pso_weighted(self, X, weights, n_clusters, n_particles, n_iterations):
        n_samples, n_features = X.shape

        # 初始化粒子群
        particles = np.random.uniform(low=np.min(X), high=np.max(X), size=(n_particles, n_clusters, n_features))

        # 初始化粒子最佳位置和全局最佳位置
        particle_best_positions = particles.copy()
        global_best_position = particles[
            np.argmin([np.sum([self.__distance_weights(weights, X[i], particle) for i in range(n_samples)]) for particle in particles])]

        # 初始化粒子速度
        velocities = np.zeros_like(particles)

        for _ in range(n_iterations):
            for i in range(n_particles):
                for j in range(n_clusters):
                    # 计算粒子位置对应的适应度值
                    inertia = np.sum([self.__distance_weights(weights, X[k], particles[i, j]) for k in range(n_samples)])
                    particle_best_inertia = np.sum(
                        [self.__distance_weights(weights, X[k], particle) for k in range(n_samples) for particle in particle_best_positions[i]])
                    global_best_inertia = np.sum(
                        [self.__distance_weights(weights, X[k], particle) for k in range(n_samples) for particle in global_best_position])

                    # 更新粒子速度和位置
                    velocities[i, j] = velocities[i, j] + np.random.rand() * (
                                particle_best_positions[i, j] - particles[i, j]) + np.random.rand() * (
                                                   global_best_position[j] - particles[i, j])
                    particles[i, j] = particles[i, j] + velocities[i, j]

                    # 限制粒子位置在数据范围内
                    particles[i, j] = np.clip(particles[i, j], np.min(X), np.max(X))

            # 更新粒子最佳位置和全局最佳位置
            for i in range(n_particles):
                if np.sum([self.__distance_weights(weights, X[k], particles[i, j]) for k in range(n_samples) for j in
                           range(n_clusters)]) < np.sum(
                        [self.__distance_weights(weights, X[k], particle) for k in range(n_samples) for particle in
                         particle_best_positions[i]]):
                    particle_best_positions[i] = particles[i].copy()

            if np.sum(
                    [self.__distance_weights(weights, X[k], particle) for k in range(n_samples) for particle in global_best_position]) > np.sum(
                    [self.__distance_weights(weights, X[k], particle) for k in range(n_samples) for particle in particle_best_positions[i]]):
                global_best_position = particle_best_positions[i].copy()

        # 根据最终的全局最佳位置进行聚类
        clusters = []
        centroids = []
        for sample in X:
            distances = [self.__distance_weights(weights, sample, centroid) for centroid in global_best_position]
            label = np.argmin(distances)
            clusters.append(label)
        for centroid in global_best_position:
            centroids.append(centroid)

        # return np.array(clusters), np.array(centroids)
        return clusters, centroids


class KMeansPSO2():
    def __distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def kmeans_pso(self, X, n_clusters, n_particles, n_iterations):
        n_samples, n_features = X.shape

        # 初始化粒子群
        particles = np.random.uniform(low=np.min(X), high=np.max(X), size=(n_particles, n_clusters, n_features))

        # 初始化粒子最佳位置和全局最佳位置
        particle_best_positions = particles.copy()
        global_best_position = particles[
            np.argmin([np.sum([self.__distance(X[i], particle) for i in range(n_samples)]) for particle in particles])]

        # 初始化粒子速度
        velocities = np.zeros_like(particles)

        for _ in range(n_iterations):
            for i in range(n_particles):
                for j in range(n_clusters):
                    # 计算粒子位置对应的适应度值
                    inertia = np.sum([self.__distance(X[k], particles[i, j]) for k in range(n_samples)])
                    particle_best_inertia = np.sum(
                        [self.__distance(X[k], particle) for k in range(n_samples) for particle in particle_best_positions[i]])
                    global_best_inertia = np.sum(
                        [self.__distance(X[k], particle) for k in range(n_samples) for particle in global_best_position])

                    # 更新粒子速度和位置
                    velocities[i, j] = velocities[i, j] + np.random.rand() * (
                            particle_best_positions[i, j] - particles[i, j]) + np.random.rand() * (
                                               global_best_position[j] - particles[i, j])
                    particles[i, j] = particles[i, j] + velocities[i, j]

                    # 限制粒子位置在数据范围内
                    particles[i, j] = np.clip(particles[i, j], np.min(X), np.max(X))

            # 更新粒子最佳位置和全局最佳位置
            for i in range(n_particles):
                if np.sum([self.__distance(X[k], particles[i, j]) for k in range(n_samples) for j in
                           range(n_clusters)]) < np.sum(
                        [self.__distance(X[k], particle) for k in range(n_samples) for particle in
                         particle_best_positions[i]]):
                    particle_best_positions[i] = particles[i].copy()

            if np.sum(
                    [self.__distance(X[k], particle) for k in range(n_samples) for particle in global_best_position]) > np.sum(
                    [self.__distance(X[k], particle) for k in range(n_samples) for particle in particle_best_positions[i]]):
                global_best_position = particle_best_positions[i].copy()

        # 根据最终的全局最佳位置进行聚类
        labels = []
        for sample in X:
            distances = [self.__distance(sample, centroid) for centroid in global_best_position]
            label = np.argmin(distances)
            labels.append(label)

        return np.array(labels)



if __name__=="__main__":
    kp = KMeansPSO()

    # 生成示例数据
    X, y = make_blobs(n_samples=100, centers=4, n_features=2, random_state=42)

    # 使用粒子群算法优化的K均值聚类算法进行聚类
    clusters, centroids = kp.kmeans_pso(X, n_clusters=4, n_particles=30, n_iterations=100)

    print("Clusters:", clusters)
    print("Centroids:", centroids)