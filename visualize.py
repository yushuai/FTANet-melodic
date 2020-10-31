import os
import math
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework import ops

from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


from constant import *
from loader import load_single_data_for_test, get_CenFreq, seq2map
from evaluator import evaluate
# from keract import display_heatmaps
# from network.msnet import create_msnet_model as create_model


os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def display_heatmaps(activations, input_image, directory):

    data_format = K.image_data_format()

    index = 0
    for layer_name, acts in activations.items():
        print(layer_name, acts.shape, end=' ')
        if acts.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue
        if len(acts.shape) <= 2:
            print('-> Skipped. 2D Activations.')
            continue
        print('')


        # computes values required to scale the activations (which will form our heat map) to be in range 0-1
        scaler = MinMaxScaler()
        # reshapes to be 2D with an automaticly calculated first dimension and second
        # dimension of 1 in order to keep scikitlearn happy
        scaler.fit(acts.reshape(-1, 1))

        # loops over each filter/neuron
        for i in range(acts.shape[-1]):
            dpi = 300
            fig = plt.figure(figsize=(input_image.shape[1]/dpi, input_image.shape[0]/dpi), dpi=dpi)
            # fig = plt.figure(figsize=(input_image.shape[1], input_image.shape[0]))
            axes = fig.add_axes([0, 0, 1, 1])
            axes.set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
        
            if len(acts.shape) == 3:
                # gets the activation of the ith layer
                if data_format == 'channels_last':
                    img = acts[0, :, i]
                elif data_format == 'channels_first':
                    img = acts[0, i, :]
                else:
                    raise Exception('Unknown data_format.')
            elif len(acts.shape) == 4:
                if data_format == 'channels_last':
                    img = acts[0, :, :, i]
                elif data_format == 'channels_first':
                    img = acts[0, i, :, :]
                else:
                    raise Exception('Unknown data_format.')
            else:
                raise Exception('Expect a tensor of 3 or 4 dimensions.')

            # scales the activation (which will form our heat map) to be in range 0-1 using
            # the previously calculated statistics
            if len(img.shape) == 1:
                img = scaler.transform(img.reshape(-1, 1))
            else:
                img = scaler.transform(img)
            # print(img.shape)
            img = Image.fromarray(img)
            # resizes the activation to be same dimensions of input_image
            img = img.resize((input_image.shape[1], input_image.shape[0]), Image.LANCZOS)
            img = np.array(img)

            # overlay the activation at 70% transparency  onto the image with a heatmap colour scheme
            # Lowest activations are dark, highest are dark red, mid are yellow
            axes.imshow(input_image / 255.0)
            axes.imshow(img, alpha=1.0, cmap='jet', interpolation='bilinear')

            # save to png
            if not os.path.exists(directory):
                os.makedirs(directory)
            output_filename = os.path.join(directory, '{}-{}_{}.png'.format(index, layer_name.split('/')[0], i))
            plt.savefig(output_filename, bbox_inches='tight', dpi=dpi, pad_inches=0)

            plt.close(fig)

        index += 1



def display_heatmap(activation, input_image, fname, alpha):
    dpi = 300
    fig = plt.figure(figsize=(input_image.shape[1]/dpi, input_image.shape[0]/dpi), dpi=dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)

    img = activation
    img = scale_minmax(img, min=0.0, max=1.0)
    img = Image.fromarray(img)
    img = img.resize((input_image.shape[1], input_image.shape[0]), Image.LANCZOS)
    img = np.array(img)

    # overlay the activation
    axes.imshow(input_image / 255.0)
    axes.imshow(img, alpha=alpha, cmap='jet', interpolation='bilinear')

    # save to png
    plt.savefig(fname, bbox_inches='tight', dpi=dpi, pad_inches=0)
    plt.close(fig)


def display_spec(activation, input_image, fname):
    dpi = 300
    fig = plt.figure(figsize=(input_image.shape[1]/dpi, input_image.shape[0]/dpi), dpi=dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    librosa.display.specshow(activation)

    #去掉边框
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    plt.savefig(fname, bbox_inches='tight', dpi=dpi, pad_inches=0)
    plt.close(fig)


def visualize(model, x, mapping, i_seg, layers):
    ## choose one segment
    data = x[i_seg]
    ground = mapping[1:, i_seg*SEG_LEN: (i_seg+1)*SEG_LEN]
    display_spec(ground, ground, 'visualization/{}_ground.png'.format(i_seg))

    ## generate image to overly
    image = data[:, :, 1] * data[:, :, 2]
    image = scale_minmax(image, 0, 255).astype(np.uint8)
    
    ## to input data
    x_in = np.expand_dims(data, axis=0)

    # visualization
    for layer_name in layers: 
        outputs = K.function([model.get_input_at(0)], [model.get_layer(layer_name).output])([x_in])
        output_mean = np.mean(outputs[0][0], axis=-1)
        # output_mean = outputs[0][0]
        print(output_mean.shape)
        display_spec(output_mean, image, 'visualization/{}_{}.png'.format(i_seg, layer_name))


def print_tf_weights(model, x, i_seg):
    data = x[i_seg]
    x_in = np.expand_dims(data, axis=0)

    print('Time Attn:')
    layers = ['conv1d_{}'.format(i) for i in range(1, 28, 4)]
    for layer in layers:
        outputs = K.function([model.get_input_at(0)], [model.get_layer(layer).output])([x_in])
        output_mean = np.mean(outputs[0][0], axis=-1)
        print(output_mean)

    print('Frequency Attn:')
    layers = ['conv1d_{}'.format(i) for i in range(3, 28, 4)]
    for layer in layers:
        outputs = K.function([model.get_input_at(0)], [model.get_layer(layer).output])([x_in])
        output_mean = np.mean(outputs[0][0], axis=-1)
        print(output_mean)


if __name__ == '__main__':
    # 1. load one audio segment
    """
    daisy1.npy
    daisy2.npy
    daisy3.npy
    daisy4.npy
    opera_fem2.npy
    opera_fem4.npy
    opera_male3.npy
    opera_male5.npy
    pop1.npy
    pop2.npy
    pop3.npy
    pop4.npy
    """
    f = 'daisy1.npy' 
    xlist, ylist = load_single_data_for_test(f, seg_len=SEG_LEN)
    CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60) # (321) #参数是特征提取时就固定的
    mapping = seq2map(ylist[0][:, 1], CenFreq) # (321, T)

    # 2. load model
    from network.ftanet_2 import create_model
    model = create_model(input_shape=IN_SHAPE)
    model.load_weights('model/ftanet_2_1015.h5')
    model.compile(loss='binary_crossentropy', optimizer=(Adam(lr=LR)))
    # model.summary()
    # layers = ['multiply_33', 'multiply_34']
    # layers = ['multiply']
    # layers.extend(['multiply_{}'.format(i) for i in range(1, 35, 5)])
    # layers.extend(['multiply_{}'.format(i) for i in range(5, 35, 5)])
    # layers = ['softmax_21']
    layers = ['reshape_30', 'reshape_31']

    # 3. visualization
    # print_tf_weights(model, xlist[0], 3)
    # visualize(model, xlist[0], mapping, 1, layers)

    data = xlist[0][5]
    for i in range(3):
        x = data[:, :, i]
        display_spec(x, x, 'visualization/input_{}.png'.format(i))

        

    # for seg in range(len(xlist[0])-1):
    #     visualize(model, xlist[0], mapping, seg, layers)
    
    # for seg in range(len(xlist[0])-1):
    #     visualize(model, xlist[0], mapping, seg, layers)
        # ## choose one segment
        # data = xlist[0][seg]
        # ground = mapping[1:, seg*SEG_LEN: (seg+1)*SEG_LEN]
        # print(ground.shape)
        # display_spec(ground, ground, 'visualization/{}_ground.png'.format(seg))

        # ## generate image to overly
        # image = data[:, :, 1] * data[:, :, 2]
        # # display_spec(image, image, 'visualization/{}_origin.png'.format(seg))
        # # image = np.log(image + 1e-9) 
        # image = scale_minmax(image, 0, 255).astype(np.uint8)
        # # image = 255 - image
        # # img = Image.fromarray(image, mode='L')
        # # img.save('visualization/origin.png', quality=95, subsampling=0)

        # ## as input data
        # x = np.expand_dims(data, axis=0)

        # # 4. visualization
        # for layer_name in layers: 
        #     outputs = K.function([model.get_input_at(0)], [model.get_layer(layer_name).output])([x])
        #     # activations = {layer_name: outputs[0]}
        #     # display_heatmaps(activations, image, 'visualization/')
        #     output_mean = np.mean(outputs[0][0], axis=-1)
        #     print(output_mean.shape)
        #     # display_heatmap(output_mean, image, 'visualization/{}.png'.format(layer_name), 0.5)
        #     display_spec(output_mean, image, 'visualization/{}_{}.png'.format(seg, layer_name))


    
    