import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
class Dkmeans:
    def __init__(self, K: int, X: np.ndarray):
        self.K = K  # số lượng cụm
        self.X = X  # ma trận tập hợp các điễm dữ liệu 

    # khởi tạo tâm cụm
    def kmeans_init_centers(self, X, k):
        # chọn ngẫu nhiên k hàng của X làm tâm ban đầu
        return X[np.random.choice(X.shape[0], k, replace=False)]

    def kmeans_assign_labels(self, X, centers):
        # tính khoảng cách giữa các điểm dữ liệu và các tâm
        D = cdist(X, centers)
        # trả về chỉ số của tâm gần nhất
        return np.argmin(D, axis=1)

    def kmeans_update_centers(self, X, labels, K):
        centers = np.zeros((K, X.shape[1]))
        for k in range(K):
            # lấy tất cả các điểm được gán vào cụm k
            Xk = X[labels == k, :]
            # lấy trung bình
            centers[k, :] = np.mean(Xk, axis=0)
        return centers

    def has_converged(self, centers, new_centers):
        # trả về True nếu hai tập hợp tâm cụm là giống nhau
        return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])

    def kmeans(self, X: np.ndarray, K: int):
        centers = [self.kmeans_init_centers(X, K)]
        labels = []
        it = 0
        while True:
            labels.append(self.kmeans_assign_labels(X, centers[-1]))
            new_centers = self.kmeans_update_centers(X, labels[-1], K)
            if self.has_converged(centers[-1], new_centers):
                break
            centers.append(new_centers)
            it += 1
        return (centers, labels, it)
    
    def visualize_clusters(X, labels, centers):
        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_points = X[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centers')
        plt.legend()
        plt.title('K-means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()