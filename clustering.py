import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics


def compute_db_scan(X, db):
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def dendrogram_initial_plot(X):
    linked = linkage(X, 'single')
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.show()


def dendrogram_get_distances(x, model, mode='l2'):
    distances = []
    weights = []
    children=model.children_
    dims = (x.shape[1],1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = x[childs[0]].reshape(dims)
        c2 = x[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1-c2)
        cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)

        x = np.vstack((x, cc.T))
        newChild_id = x.shape[0]-1

        if mode == 'l2':  # Increase the higher level cluster size suing an l2 norm
            added_dist = (c1Dist**2 + c2Dist**2)**0.5
            dNew = (d**2 + added_dist**2)**0.5
        elif mode == 'max':  # If the previrous clusters had higher distance, use that one
            dNew = max(d, c1Dist ,c2Dist)
        elif mode == 'actual':  # Plot the actual distance.
            dNew = d

        wNew = (c1W + c2W)
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew

        distances.append(dNew)
        weights.append(wNew)
    return distances, weights


def dendrogram_func(x, model):
    distance, weight = dendrogram_get_distances(x, model)
    linkage_matrix = np.column_stack([model.children_, distance, weight]).astype(float)
    plt.figure()
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(linkage_matrix)
    plt.show()


# def dendrogram_func1(model, points):
#     plt.figure()
#     model = model.fit(points)
#     plt.title('Hierarchical Clustering Dendrogram')
#     plot_dendrogram1(model, truncate_mode='level', p=3)
#     plt.xlabel("Number of points in node")
#     plt.show()


# def plot_dendrogram1(model, **kwargs):
#
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for j, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[j] = current_count
#
#     linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
#     dendrogram(linkage_matrix, **kwargs)


def kmeans_clusters_plot(km, x):
    plt.figure()
    y_kmeans = km.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = km.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


def elbow_plot(x):
    distortions = []
    for j in range(1, 20):
        km = KMeans(n_clusters=j, init='random')
        km.fit(x)
        kmeans_clusters_plot(km, x)
        distortions.append(km.inertia_)

    plt.figure()
    plt.plot(range(1, 20), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


def k_means(data_set, points):
    all_rss = []
    k_list = []
    for k in range(1, 10, 1):
        print(k)
        k_list.append(k)
        centroids = []
        for i in range(k):
            centroids.append(points[random.randint(-1, len(data_set) - 1)])

        for m in range(50):
            clusters = {}
            rss = 0
            for i in range(len(data_set)):
                temp = []
                for j in centroids:
                    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(points[i], j)]))
                    temp.append(distance)
                for j in range(k):
                    if temp[j] == min(temp):
                        clusters[i] = j
                        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(points[i], centroids[j])]))
                        rss = rss + distance
                        break

            # change centroids
            all_new_centroids = []
            for j in range(k):
                new_centroid = [0 for _ in range(2)]
                points_in_cluster = 0
                for l in range(len(data_set)):
                    if clusters[l] == j:
                        points_in_cluster = points_in_cluster + 1
                        new_centroid = [new_centroid[i] + points[l][i] for i in range(2)]
                new_centroid = [new_centroid[i] / points_in_cluster for i in range(2)]
                all_new_centroids.append(new_centroid)
            centroids = all_new_centroids

        all_rss.append(rss)

    print("min rss:", min(all_rss))
    print("best k from 1 to 30 is:", 1 + all_rss.index(min(all_rss)))
    plt.plot(k_list, all_rss)


# read from file
data_set1 = pd.read_csv("Dataset1.csv")
data_set2 = pd.read_csv("Dataset2.csv")

points1 = []
for i in range(len(data_set1)):
    points1.append([data_set1.iloc[i][0], data_set1.iloc[i][1]])

points2 = []
for i in range(len(data_set2)):
    points2.append([data_set2.iloc[i][0], data_set2.iloc[i][1]])

points1 = np.array(points1)
points2 = np.array(points2)

elbow_plot(points1)
elbow_plot(points2)

clustering_db1 = DBSCAN(eps=0.2, min_samples=5).fit(points1)
clustering_db2 = DBSCAN(eps=2, min_samples=5).fit(points2)

print("data set1:")
compute_db_scan(points1, clustering_db1)

print("data set2:")
compute_db_scan(points2, clustering_db2)


dendrogram_initial_plot(points1)
dendrogram_initial_plot(points2)

Model = AgglomerativeClustering(n_clusters=2, linkage="ward")
Model.fit(points1)
dendrogram_func(points1, Model)

Model = AgglomerativeClustering(n_clusters=2, linkage="ward")
Model.fit(points2)
dendrogram_func(points2, Model)


# # setting distance_threshold=0 ensures we compute the full tree
# m = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
# dendrogram_func1(m, points1)
# dendrogram_func1(m, points2)
