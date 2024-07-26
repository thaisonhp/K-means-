import matplotlib.pyplot as plt
import numpy as np
def kmeans_display(X, label):
    K = np.amax(label) + 1
    for k in range(K):
        Xk = X[label == k, :]
        plt.plot(Xk[:, 0], Xk[:, 1], 'o', markersize=4, alpha=0.8, label=f'Cluster {k}')
    
    plt.axis('equal')
    plt.legend()
    plt.show()
    