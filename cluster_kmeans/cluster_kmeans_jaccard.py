import numpy as np
import statistics
from sklearn import metrics
from cluster_kmeans.kmeans_straight import kmeans

def calc_jaccard(cluster1, cluster2):
    # jaccard = metrics.jaccard_score(cluster1, cluster2, average='macro')
    jaccard = metrics.jaccard_score(cluster1, cluster2, average='weighted')
    # jaccard = metrics.jaccard_score(cluster1, cluster2, average='micro')
    # jaccard = 0
    # print("cluster1:", cluster1)
    # print("cluster2:", cluster2)
    return jaccard

def calc_jaccard_mean(list_cc):
    visited = set()  # 用一个集合来记录已经访问过的元素的索引对
    i_sum = 0
    avg_jaccard = 0
    list_jaccard = list()
    for i in range(len(list_cc)):
        for j in range(len(list_cc)):
            if i == j:
                continue
            if (i, j) in visited or (j, i) in visited:
                continue  # 如果索引对已经访问过，则跳过本次循环

            visited.add((i, j))  # 将索引对添加到已访问集合中
            #
            jaccard_i = calc_jaccard(list_cc[i], list_cc[j])
            list_jaccard.append(jaccard_i)
            avg_jaccard += jaccard_i
            i_sum += 1
            # 在这里可以对a[i]和b[j]执行相应的操作
            # print(f"a[{i}] = {list_cc[i]}, b[{j}] = {list_cc[j]}")
            # print("i=%.2d, j=%.2d - jaccard_i = %.6f" % (i , j, jaccard_i))
    # print("avg_jaccard=%.6f, i_sum=%d" % (avg_jaccard, i_sum))
    avg_jaccard /= i_sum
    # print("avg_jaccard=%.6f" % (avg_jaccard))
    print("jaccard-max=%.6f, min=%.6f, std=%.6f" % (max(list_jaccard), min(list_jaccard), (statistics.stdev(list_jaccard))))
    return avg_jaccard



if __name__ == '__main__':
    # list_cc = [[1, 2, 3, 2, 1, 2], [1, 1, 3, 2, 2, 1], [1, 2, 3, 1, 1, 2]]
    list_cc = [[0.1, 0.2, 0.3, 0.2, 0.1, 0.2], [0.1, 0.1, 0.3, 0.2, 0.2, 0.1], [0.1, 0.2, 0.3, 0.1, 0.1, 0.2]]
    # list_cc = [[0.1, 0.2, 0.3, 0.2, 0.1, 0.2], [0.1, 0.1, 0.3, 0.2, 0.2, 0.1], [0.1, 0.2, 0.3, 0.1, 0.1, 0.2]]

    result = [[int(num * 100) for num in sublist] for sublist in list_cc]

    print("结果:", result)
    calc_jaccard_mean(result)