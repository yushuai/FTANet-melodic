import os
import argparse
import time
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy

from constant import *
from generator import create_data_generator
from loader import load_data, load_data_for_test #TODO
from evaluator import evaluate

from network.ftanet import create_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 指定输出的模型的路径
parser = argparse.ArgumentParser()
parser.add_argument("model_file", type=str, help="model file")
checkpoint_model_file = parser.parse_args().model_file


# log_file_name = 'log/log-train-{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))
log_file_name = checkpoint_model_file.replace('model/', 'log/').replace('.h5', '.log')
log_file = open(log_file_name, 'wb')

# 日志函数
def log(message):
    message_bytes = '{}\n'.format(message).encode(encoding='utf-8')
    log_file.write(message_bytes)
    print(message)


## 文件列表
train_list_file = '/data1/project/MCDNN/data/train_npy.txt'
valid_list_file = '/data1/project/MCDNN/data/valid_npy.txt'


##--- 加载数据 ---##
# x: (n, freq_bins, time_frames, 3) extract from audio by cfp_process
# y: (n, freq_bins+1, time_frames) from ground-truth
train_x, train_y, train_num = load_data(train_list_file, seg_len=SEG_LEN)
log('Loaded {} segments from {}.'.format(train_num, train_list_file))
valid_x, valid_y = load_data_for_test(valid_list_file, seg_len=SEG_LEN)
log('Loaded features for {} files from {}.'.format(len(valid_y), valid_list_file))

##--- Data Generator ---##
log('\nCreating generators...')
train_generator = create_data_generator(train_x, train_y, batch_size=BATCH_SIZE)

##--- 网络 ---##
log('\nCreating model...')

model = create_model(input_shape=IN_SHAPE)
model.compile(loss='binary_crossentropy', optimizer=(Adam(lr=LR)))
# model.summary()

##--- 开始训练 ---##
log('\nTaining...')
log('params={}'.format(model.count_params()))

epoch, iteration = 0, 0
best_OA, best_epoch = 0, 0
mean_loss = 0
time_start = time.time()
while epoch < EPOCHS:
    iteration += 1
    print('Epoch {}/{} - {:3d}/{:3d}'.format(
        epoch+1, EPOCHS, iteration%(train_num//BATCH_SIZE), train_num//BATCH_SIZE), end='\r')
    # 取1个batch数据
    X, y = next(train_generator)
    # 训练1个iteration
    loss = model.train_on_batch(X, y)
    mean_loss += loss
    # 每个epoch输出信息
    if iteration % (train_num//BATCH_SIZE) == 0:
        ## train meassage
        epoch += 1
        traintime  = time.time() - time_start
        mean_loss /= train_num//BATCH_SIZE
        print('', end='\r')
        log('Epoch {}/{} - {:.1f}s - loss {:.4f}'.format(epoch, EPOCHS, traintime, mean_loss))
        ## valid results
        avg_eval_arr = evaluate(model, valid_x, valid_y, BATCH_SIZE)
        # save to model
        if avg_eval_arr[-1] > best_OA:
            best_OA = avg_eval_arr[-1]
            best_epoch = epoch
            model.save_weights(checkpoint_model_file)
            log('Saved to ' + checkpoint_model_file)
        log('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}% BestOA {:.2f}%'.format(
            avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4], best_OA))
        # early stopping
        if epoch - best_epoch >= PATIENCE:
            log('Early stopping with best OA {:.2f}%'.format(best_OA))
            break
        ## initialization
        mean_loss = 0
        time_start = time.time()

