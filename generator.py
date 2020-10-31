import math
import random
import numpy as np

from tensorflow.keras.utils import to_categorical


def std_normalize(data): 
    # normalize as 64 bit, to avoid numpy warnings
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    data = data.copy() - mean
    if std != 0.:
        data = data / std
    return data.astype(np.float32)


def create_data_generator(features, labels, batch_size):
    
    length = len(labels)
    idx = [i for i in range(length)]
    random.shuffle(idx)

    i = 0
    while True:
        # 每个epoch后进行shuffle
        if i + batch_size > length:
            i = 0
            random.shuffle(idx)

        # 每次取batch_size个key
        X, y = [], []
        for j in range(i, i + batch_size):
            # 对每一个，取feature
            feature = features[idx[j]]
            label = labels[idx[j]]
            # # 对数据进行标准化 # 可能引起一些问题
            # feature = std_normalize(feature)
            # 存入集合
            X.append(feature)
            y.append(label)

        i += batch_size
        yield np.stack(X, axis=0), np.stack(y, axis=0)

