import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from constant import *
from loader import get_CenFreq, load_single_data_for_test
from evaluator import iseg, est


## 获得某一首歌预测结果
def get_est(model, fname):
    # 载入一个数据
    x_list, y_list = load_single_data_for_test(fname, seg_len=SEG_LEN)
    x = x_list[0]
    y = y_list[0]
    
    # 预测并拼接
    num = x.shape[0] // BATCH_SIZE
    if x.shape[0] % BATCH_SIZE != 0:
        num += 1
    preds = []
    for j in range(num):
        # x: (batch_size, freq_bins, seg_len)
        if j == num - 1:
            X = x[j*BATCH_SIZE : ]
            length = x.shape[0]-j*BATCH_SIZE
        else:
            X = x[j*BATCH_SIZE : (j+1)*BATCH_SIZE]
            length = BATCH_SIZE

        prediction = model.predict(X, length)
        preds.append(prediction)

    # (num*bs, freq_bins, seg_len) to (freq_bins, T)
    preds = np.concatenate(preds, axis=0)
    preds = iseg(preds)

    # ground-truth
    ref_arr = y
    time_arr = y[:, 0]
    
    # trnasform to f0ref
    CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
    est_arr = est(preds, CenFreq, time_arr)

    return est_arr


if __name__ == '__main__':
    # 指定测试哪个模型
    from network.tfsknet0904 import create_tfskent_model as create_model
    model_file = 'model/tfsknet0904_0904_2.h5'
    
    # 载入模型
    model = create_model(input_shape=IN_SHAPE)
    model.load_weights(model_file)

    # 选择一首歌
    fname = 'opera_male5.npy'
    est_f0ref = get_est(model, fname)

    # 保存文件
    np.savetxt('est_f0ref/' + fname.replace('.npy', '.txt'), est_f0ref, fmt="%.3f", delimiter=" ")
