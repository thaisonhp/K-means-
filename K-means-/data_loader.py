import numpy as np 
import pandas as pd
from numpy import random
import os
import json
from urllib import request, parse, error
import certifi
import ssl

TEST_CASES = {
    53: {
        'name': 'Iris',
        'n_cluster': 3,
        'test_points': [5.1, 3.5, 1.4, 0.2]
    },
    109: {
        'name': 'Wine',
        'n_cluster': 4,
        'test_points': [14.23, 1.71, 2.43, 15.6,
                        127, 2.80, 3.06, 0.28,
                        2.29, 5.64, 1.04, 3.92,
                        1065]
    },
    602: {
        'name': 'DryBean',
        'n_cluster': 7,
        'test_points': [
            28395, 610.291, 208.178117, 173.888747,
            1.197191, 0.549812, 28715, 190.141097,
            0.763923, 0.988856, 0.958027, 0.913358,
            0.007332, 0.003147, 0.834222, 0.998724]
    }
}


def round_float(number: float) -> float: # -> chỉ định kiểu dữ liệu trả về
    '''
      hàm này để làm tròn một số thực đến 3 chữ số thập phân.
      đàu vào: number (số thực)
      đầu ra: number (số thực làm tròn)
    '''
    return round(number, 3)


def euclidean_distances(A: np.ndarray, B: np.ndarray, axis: int = None):
  '''
    hàm để tính khoảng cách eculid
    đầu vào : toạ độ 2 điểm A , B
    đầu ra : khoảng cách euclid - norm 2.
  '''
  return np.linalg.norm(A - B, axis=axis)

def load_dataset(data: dict, file_csv: str = ''):
    '''
    hàm để load dữ liệu
    đầu vào : dictionary data ( dữ liệu) và dường dẫn file csv
    đầu ra : ma trận đặc trưng X
    '''
    # label_name = data['data']['target_col']
    print('uci_id=', data['data']['uci_id'])  # Mã bộ dữ liệu
    print('data name=', data['data']['name'])  # Tên bộ dữ liệu
    print('data abstract=', data['data']['abstract'])  # Tên bộ dữ liệu
    print('feature types=', data['data']['feature_types'])  # Kiểu nhãn
    print('num instances=', data['data']['num_instances'])  # Số lượng điểm dữ liệu
    print('num features=', data['data']['num_features'])  # Số lượng đặc trưng
    # from pprint import pprint
    metadata = data['data']
    # colnames = ['Area', 'Perimeter']
    # df = pd.read_csv(metadata['data_url'], names=colnames, header=None)
    df = pd.read_csv(file_csv if file_csv != '' else metadata['data_url'], header=0)
    print('data top', df.head())  # Hiển thị một số dòng dữ liệu
    # Trích xuất ma trận đặc trưng X (loại trừ nhãn lớp)
    return df.iloc[:, :-1].values


# Lấy dữ liệu từ ISC UCI (53: Iris, 602: DryBean, 109: Wine)
def fetch_data_from_uci(name_or_id=53) -> dict:
    api_url = 'https://archive.ics.uci.edu/api/dataset' # đường dẫn truy cập đến API của bộ dữ liệu
    # kiểm tra đầu vào là ID (số) hay name (str)
    if isinstance(name_or_id, str): # nếu nó là chuỗi (name)
        api_url += '?name=' + parse.quote(name_or_id) # mã hoá chuỗi name đó để trở thành một định dạngg an toàn cho URL
    else:
        api_url += '?id=' + str(name_or_id)
    try:
        '''
        Tạo một ngữ cảnh SSL an toàn với chứng chỉ CA đáng tin cậy.
        Sử dụng ngữ cảnh SSL này để mở URL api_url.
        Trả về đối tượng phản hồi (response object) từ URL.
        '''
        response = request.urlopen(api_url, context=ssl.create_default_context(cafile=certifi.where()))
        '''
        2 tham số là :
        api_url :
        context : tạo ngữ cảnh SSL để mở đường dẫn URL
        '''
        return {'X': load_dataset(data=json.load(response))}
    except (error.URLError, error.HTTPError):
        raise ConnectionError('Error connecting to server')