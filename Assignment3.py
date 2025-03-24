import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
file_path = r"C:\Users\KIIT\Downloads\kmeans - kmeans_blobs (1).csv"
data_frame = pd.read_csv(file_path)
def scale_data(dataset):
    return (dataset - dataset.min()) / (dataset.max() - dataset.min())
data_points = data_frame.values[:, :2]
data_points = scale_data(data_points)
def kmeans_clustering(points, num_clusters, max_iterations=100, tolerance=1e-4):
    np.random.seed(42)
    cluster_centers = points[np.random.choice(len(points), num_clusters, replace=False)]
    for _ in range(max_iterations):
        distances = np.linalg.norm(points[:, np.newaxis] - cluster_centers, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        new_centers = np.array([points[cluster_assignments == i].mean(axis=0) for i in range(num_clusters)])
        if np.linalg.norm(new_centers - cluster_centers) < tolerance:
            break
        cluster_centers = new_centers
    return cluster_assignments, cluster_centers
labels_2, centers_2 = kmeans_clustering(data_points, num_clusters=2)
labels_3, centers_3 = kmeans_clustering(data_points, num_clusters=3)
def visualize_clusters(points, labels, centers, num_clusters, title):
    plt.figure(figsize=(6, 5))
    for i in range(num_clusters):
        plt.scatter(points[labels == i][:, 0], points[labels == i][:, 1], label=f"Cluster {i+1}")
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()
visualize_clusters(data_points, labels_2, centers_2, num_clusters=2, title="K-Means Clustering (k=2)")
visualize_clusters(data_points, labels_3, centers_3, num_clusters=3, title="K-Means Clustering (k=3)")
