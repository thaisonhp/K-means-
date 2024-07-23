import data_loader
import untities
import numpy as np 
import k_means
TEST_CASES = {
    53: {
        'name': 'Iris',
        'n_cluster': 3,
        'test_points': [5.1, 3.5, 1.4, 0.2]
    },
    109: {
        'name': 'Wine',
        'n_cluster': 4,
        'test_points': [14.23, 1.71, 2.43, 15.6, 127, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]
    },
    602: {
        'name': 'DryBean',
        'n_cluster': 7,
        'test_points': [
            28395, 610.291, 208.178117, 173.888747, 1.197191, 0.549812, 28715, 190.141097, 0.763923, 0.988856,
            0.958027, 0.913358, 0.007332, 0.003147, 0.834222, 0.998724]
    }
}

if __name__ == "__main__":
    import time
    _start_time = time.time()
    DATA_ID = 53  # 53: Iris, 109: Wine, 602: DryBean
    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _dt = data_loader.fetch_data_from_uci(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round(time.time() - _start_time, 2))
        # visualize dữ liệu
        data_points = len(_dt['X'])
        n_cluster = _TEST['n_cluster'] # Số lượng cụm
        N = int(data_points / n_cluster)   # số điểm dữ liệu trong từng cụm 
        original_label = np.asarray([0]*N + [1]*N + [2]*N)
        untities.kmeans_display(_dt['X'], original_label)
        # Chạy thuật toán k-means
        kmeans_model = k_means.Dkmeans(X=_dt['X'], K=n_cluster)
        (centers, labels, it) = kmeans_model.kmeans(X=_dt['X'], K=n_cluster)
        # visulize dữ liệu sau khi phân cụm  
        k_means.Dkmeans.visualize_clusters(X=_dt['X'],labels = labels[-1] ,centers=centers[-1])
        print(f'Số lần lặp: {it}')
        print('Tâm cụm cuối cùng:', centers[-1])