import os
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from constant import *
from loader import load_data_for_test
from evaluator import evaluate

from network.ftanet import create_model

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

log_file_name = 'log/log-test-{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))
log_file = open(log_file_name, 'wb')

def log(message):
    message_bytes = '{}\n'.format(message).encode(encoding='utf-8')
    log_file.write(message_bytes)
    print(message)


## 测试函数
def test(list_file, model):
    log('\n======'+list_file.split('/')[-1].replace('_npy.txt', '')+'======')

    ##--- 加载数据 ---##
    xlist, ylist = load_data_for_test(list_file, seg_len=SEG_LEN)
    log('Loaded features for {} files from {}.'.format(len(ylist), list_file))

    ##--- 测试---##
    log('Testing...')
    avg_eval_arr = evaluate(model, xlist, ylist, BATCH_SIZE)
    log('\nVR: {:.2f}%\nVFA: {:.2f}%\nRPA: {:.2f}%\nRCA: {:.2f}%\nOA: {:.2f}%'.format(
        avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4]))


def test_song(list_file, model):
    with open(list_file) as f:
        feature_files = f.readlines()
        for f in feature_files:
            log('\n' + f.replace('\n', ''))
            xlist, ylist = load_single_data_for_test(f, seg_len=SEG_LEN)

            ##--- 测试---##
            avg_eval_arr = evaluate(model, xlist, ylist, BATCH_SIZE)
            log('VR: {:.2f}%  VFA: {:.2f}%  RPA: {:.2f}%  RCA: {:.2f}%  OA: {:.2f}%'.format(
                avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4]))


## 文件列表
folder = '/data1/project/MCDNN/data'
test_list_files = [
    # 'train_npy.txt',
    'valid_npy.txt',
    # 'test_02_npy.txt',
    # 'test_03_npy.txt',
    # 'test_04_npy.txt',
    'test_05_npy.txt',
    'test_06_npy.txt'
]


# 指定测试哪个模型
parser = argparse.ArgumentParser()
parser.add_argument("model_file", type=str, help="model file")
model_file = parser.parse_args().model_file
# model = load_model(model_file)
model = create_model(input_shape=IN_SHAPE)
model.load_weights(model_file)

model.summary()
# stringlist = []
# model.summary(print_fn=lambda x: stringlist.append(x))
# short_model_summary = "\n".join(stringlist)
# log(short_model_summary)


# test_song(os.path.join(folder, 'test_06_npy.txt'), model)
# exit()

for i in range(len(test_list_files)):
    test(os.path.join(folder, test_list_files[i]), model)
    
