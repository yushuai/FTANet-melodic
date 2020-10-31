## without selective fusion
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Lambda, \
    GlobalAveragePooling2D, Activation, MaxPooling2D, AveragePooling2D, \
    Concatenate, Add, Multiply, Softmax, Reshape, UpSampling2D, Permute, Conv1D


def FTA_Module(x, shape, kt, kf):
    x = BatchNormalization()(x)

    ## Residual
    x_r = Conv2D(shape[2], (1, 1), padding='same', activation='relu')(x)

    ## Time Attention
    # Attn Map (1, T, C), FC
    a_t = Lambda(K.mean, arguments={'axis': -3})(x)
    a_t = Conv1D(shape[2], kt, padding='same', activation='selu')(a_t)
    a_t = Conv1D(shape[2], kt, padding='same', activation='selu')(a_t) #2
    a_t = Softmax(axis=-2)(a_t)
    a_t = Reshape((1, shape[1], shape[2]))(a_t)
    # Reweight
    x_t = Conv2D(shape[2], (3, 3), padding='same', activation='selu')(x)
    x_t = Conv2D(shape[2], (5, 5), padding='same', activation='selu')(x_t)
    x_t = Multiply()([x_t, a_t])
    ###TODO:参考Strip pooling，将两个map相乘后再使用

    # Frequency Attention
    # Attn Map (F, 1, C), Conv1D
    a_f = Lambda(K.mean, arguments={'axis': -2})(x)
    a_f = Conv1D(shape[2], kf, padding='same', activation='selu')(a_f)
    a_f = Conv1D(shape[2], kf, padding='same', activation='selu')(a_f)
    a_f = Softmax(axis=-2)(a_f)
    a_f = Reshape((shape[0], 1, shape[2]))(a_f)
    # Reweight
    x_f = Conv2D(shape[2], (3, 3), padding='same', activation='selu')(x)
    x_f = Conv2D(shape[2], (5, 5), padding='same', activation='selu')(x_f)
    x_f = Multiply()([x_f, a_f])

    x = Add()([x_r, x_t, x_f])
    return x


def create_model(input_shape=(320, 430, 3)):
    visible = Input(shape=input_shape)
    x = BatchNormalization()(visible)

    ## Bottom
    # bm = BatchNormalization()(x)
    bm = x
    bm = Conv2D(16, (4, 1), padding='valid', strides=(4, 1), activation='selu')(bm) # 80
    bm = Conv2D(16, (4, 1), padding='valid', strides=(4, 1), activation='selu')(bm) # 20
    bm = Conv2D(16, (4, 1), padding='valid', strides=(4, 1), activation='selu')(bm) # 5
    bm = Conv2D(1,  (5, 1), padding='valid', strides=(5, 1), activation='selu')(bm) # 1

    # 保持高分辨率，关注细节
    shape=input_shape
    x_r, x_t, x_f = FTA_Module(x, (shape[0], shape[1], 32), 3, 3)
    x = Add()([x_r, x_t, x_f])
    x = MaxPooling2D((2, 2))(x)

    x_r, x_t, x_f = FTA_Module(x, (shape[0]//2, shape[1]//2, 64), 3, 3)
    x = Add()([x_r, x_t, x_f])
    x = MaxPooling2D((2, 2))(x)

    x_r, x_t, x_f = FTA_Module(x, (shape[0]//4, shape[1]//4, 128), 3, 3)
    x = Add()([x_r, x_t, x_f])

    x_r, x_t, x_f = FTA_Module(x, (shape[0]//4, shape[1]//4, 128), 3, 3)
    x = Add()([x_r, x_t, x_f])

    x = UpSampling2D((2, 2))(x)
    x_r, x_t, x_f = FTA_Module(x, (shape[0]//2, shape[1]//2, 64), 3, 3)
    x = Add()([x_r, x_t, x_f])

    x = UpSampling2D((2, 2))(x)
    x_r, x_t, x_f = FTA_Module(x, (shape[0], shape[1], 32), 3, 3)
    x = Add()([x_r, x_t, x_f])
    
    x_r, x_t, x_f = FTA_Module(x, (shape[0], shape[1], 1), 3, 3)
    x = Add()([x_r, x_t, x_f])
    x = Concatenate(axis=1)([bm, x])
   
    # Softmax
    x = Lambda(K.squeeze, arguments={'axis': -1})(x) # (321, 430)
    x = Softmax(axis=-2)(x)

    return Model(inputs=visible, outputs=x)