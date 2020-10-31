import numpy as np
import numpy
from numpy.core.fromnumeric import std
import mir_eval

from loader import get_CenFreq


def std_normalize(data): 
    # normalize as 64 bit, to avoid numpy warnings
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    data = data.copy() - mean
    if std != 0.:
        data = data / std
    return data.astype(np.float32)


def est(output, CenFreq, time_arr):
    # output: (freq_bins, T)
    CenFreq[0] = 0
    est_time = time_arr
    est_freq = np.argmax(output, axis=0)

    for j in range(len(est_freq)):
        est_freq[j] = CenFreq[int(est_freq[j])]

    if len(est_freq) != len(est_time):
        new_length = min(len(est_freq), len(est_time))
        est_freq = est_freq[:new_length]
        est_time = est_time[:new_length]

    est_arr = np.concatenate((est_time[:, None], est_freq[:, None]), axis=1)

    return est_arr


def melody_eval(ref, est):
    ref_time = ref[:, 0]
    ref_freq = ref[:, 1]

    est_time = est[:, 0]
    est_freq = est[:, 1]

    output_eval = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)
    VR = output_eval['Voicing Recall'] * 100.0
    VFA = output_eval['Voicing False Alarm'] * 100.0
    RPA = output_eval['Raw Pitch Accuracy'] * 100.0
    RCA = output_eval['Raw Chroma Accuracy'] * 100.0
    OA = output_eval['Overall Accuracy'] * 100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr


def iseg(data):
    # data: (batch_size, freq_bins, seg_len)
    new_length = data.shape[0] * data.shape[-1] # T = batch_size * seg_len
    new_data = np.zeros((data.shape[1], new_length)) # (freq_bins, T)
    for i in range(len(data)):
        new_data[:, i * data.shape[-1] : (i + 1) * data.shape[-1]] = data[i]
    return new_data


def evaluate(model, x_list, y_list, batch_size):
    avg_eval_arr = np.array([0, 0, 0, 0, 0], dtype='float64')
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        
        # predict and concat
        num = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            num += 1
        preds = []
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                X = x[j*batch_size : ]
                length = x.shape[0]-j*batch_size
            else:
                X = x[j*batch_size : (j+1)*batch_size]
                length = batch_size

            # for k in range(length): # normalization
            #     X[k] = std_normalize(X[k]) 
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

        # evaluate
        eval_arr = melody_eval(ref_arr, est_arr)
        avg_eval_arr += eval_arr
    
    avg_eval_arr /= len(x_list)
    # VR, VFA, RPA, RCA, OA
    return avg_eval_arr


# Just for test
if __name__ == '__main__':
    import os
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import load_model
    from tensorflow.keras.metrics import categorical_accuracy
    from loader import load_data_for_test, load_data
    # from train import acc
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    def acc(y_true, y_pred):
        y_true = K.permute_dimensions(y_true, (0, 2, 1))
        y_pred = K.permute_dimensions(y_pred, (0, 2, 1))
        return categorical_accuracy(y_true, y_pred)
        # return K.cast(K.equal(K.argmax(y_true, axis=-2), K.argmax(y_pred, axis=-2)), K.floatx())

    avg_eval_arr_rp = np.array([0, 0, 0, 0, 0], dtype='float64')
    avg_eval_arr_rl = np.array([0, 0, 0, 0, 0], dtype='float64')
    avg_eval_arr_lp = np.array([0, 0, 0, 0, 0], dtype='float64')

    # x_list, y_list = load_data_for_test('/data1/project/MCDNN/data/test_02_npy.txt') #Okay
    # x_temp, y_temp, _ = load_data('/data1/project/MCDNN/data/test_02_npy.txt') #Okay
    x_list, y_list = load_data_for_test('/data1/project/MCDNN/data/train_npy.txt')
    x_temp, y_temp, _ = load_data('/data1/project/MCDNN/data/train_npy.txt')
    model = load_model('model/msnet_0805.h5', compile=False)
    batch_size = 8

    # y_temp = np.array(y_temp)
    idx_st = 0
    print(len(x_list), len(y_list))
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]

        # predict and concat
        num = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            num += 1
        preds_raw = []
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                X = x[j*batch_size : ]
                batch_x = x_temp[idx_st+j*batch_size : idx_st+x.shape[0]]
                batch_y = y_temp[idx_st+j*batch_size : idx_st+x.shape[0]]
                length = x.shape[0]-j*batch_size
            else:
                X = x[j*batch_size : (j+1)*batch_size]
                batch_x = x_temp[idx_st+j*batch_size : idx_st+(j+1)*batch_size]
                batch_y = y_temp[idx_st+j*batch_size : idx_st+(j+1)*batch_size]
                length = batch_size
            # X_normed = std_normalize(X)
            prediction = model.predict(X, length)
            # print(np.shape(X), np.shape(batch_x))
            print('train-test', K.eval(K.mean(K.equal(np.array(X), np.array(batch_x)))), end=' ')
            print('acc', K.eval(K.mean(acc(np.array(batch_y), prediction))))
            preds_raw.append(prediction)

        preds_raw = np.concatenate(preds_raw, axis=0) ###
        print('preds', preds_raw.shape, end='; ')
        preds = iseg(preds_raw)
        print(preds.shape, end='; ')


        # train labels
        labels_raw = y_temp[idx_st : idx_st+x.shape[0]]
        idx_st += x.shape[0]

        print('labels', np.shape(labels_raw), end='; ')
        labels = iseg(np.array(labels_raw))
        print(labels.shape, end=' ')

        # print('acc', K.eval(K.mean(acc(np.array(labels_raw), preds_raw))), end='; ')


        # ground-truth
        ref_arr = y
        time_arr = y[:, 0]
        print('ground-truth', len(time_arr))
        
        # trnasform to f0ref
        CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
        est_arr_pred = est(preds, CenFreq, time_arr)
        est_arr_label = est(labels, CenFreq, time_arr)

        # cnt = 0
        # for i in range(min(np.shape(est_arr)[0], np.shape(ref_arr)[0])):
        #     # print(i, est_arr[i][1], ref_arr[i][1])
        #     if abs(est_arr[i][1] - ref_arr[i][1])>1:
        #         cnt += 1
        #         # print(i, est_arr[i][1], ref_arr[i][1])
        # print(cnt)

        # evaluate
        avg_eval_arr_rp += melody_eval(ref_arr, est_arr_pred)
        avg_eval_arr_rl += melody_eval(ref_arr, est_arr_label)
        avg_eval_arr_lp += melody_eval(est_arr_label, est_arr_pred)
    
    avg_eval_arr_rp /= len(x_list)
    avg_eval_arr_rl /= len(x_list)
    avg_eval_arr_lp /= len(x_list)
    print(avg_eval_arr_rp)
    print(avg_eval_arr_rl)
    print(avg_eval_arr_lp)
