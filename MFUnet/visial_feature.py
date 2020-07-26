from myconv_keras import build_mycov, build_myconv01, build_myconv03, build_mycov11, build_mycov12
from image_util import ImageDataProvider
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
from keras.utils import plot_model
import os
from helper_functions import *
import matplotlib.pyplot as plt
import scipy.io as scio
import matplotlib

builder = 'myconv12'

if builder == 'myconv':
    model_name = "myconv"
    model = build_mycov(input_shape=[512, 512, 3], first_filters=32)
    model_path = "./orangeNet_trained/Gland/model/myconv/"

if builder == 'myconv11':
    # 0.5332
    model_name = "myconv11"
    model = build_mycov11(input_shape=[512, 512, 3], first_filters=32)
    model_path = "./orangeNet_trained/Gland/model/myconv11/"

if builder == 'myconv12':
    # miou54
    model_name = "myconv12"
    model = build_mycov12(input_shape=[512, 512, 3], first_filters=32)
    model_path = "./orangeNet_trained/Gland/model/myconv12/"

if builder == 'myconv01':
    model_name = "myconv01"
    model = build_myconv01(512, 512, 3)
    model_path = "./orangeNet_trained/Gland/model/myconv01/"

if builder == 'myconv02':
    model_name = builder
    model = build_myconv01(512, 512, 3, filter_rate=2)
    model_path = "./orangeNet_trained/Gland/model/myconv02/"

if builder == 'myconv03':
    model_name = builder
    model = build_myconv03(512, 512, 3, filter_rate=1)
    model_path = "./orangeNet_trained/Gland/model/myconv03/"

if builder == 'unet':
    model_name = 'unet'
    model_path = "./orangeNet_trained/Gland/model/unet/"
    model = U_Net(512, 512, 3)

if builder == 'unet++':
    model_name = 'unetPlus'
    model_path = "./orangeNet_trained/Gland/model/unetPlus/"
    model = Nest_Net(512, 512, 3)


model.load_weights("single_gpu_model.h5")
batch_size = 2

generator = ImageDataProvider("../data/Gland/verify/*.tif", shuffle_data=False)
x_test, y_test = generator(8)
print(">> Test  data: {} | {} ~ {}".format(x_test.shape, np.min(x_test), np.max(x_test)))
print(">> Test  mask: {} | {} ~ {}\n".format(y_test.shape, np.min(y_test), np.max(y_test)))
for i in range(4):
    i += 2
    for j in range(2):
        j += 1
        for k in range(2):
            if k == 1:
                relu_name = "relu5_5"
            else:
                relu_name = "relu3_3"
            vision_model = Model(inputs=model.input,
                                 outputs=model.get_layer("stage"+str(i)+"_unit"+str(j)+"_"+relu_name).output)

            p_test = vision_model.predict(x_test, batch_size=batch_size, verbose=1)
            print("stage"+str(i)+"_unit"+str(j)+relu_name+"shape:", p_test.shape)
            scio.savemat('./visal/stage'+str(i)+'_unit'+str(j)+'/'+relu_name + '.mat',
                         {'data': x_test, 'label': y_test, 'prediction': p_test})

