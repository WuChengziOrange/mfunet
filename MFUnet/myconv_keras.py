import keras.backend as K
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Dense
from keras.models import Model
from keras.engine import get_source_inputs
from my_blocks import SK_change_block, SK_block_deepwise
from my_blocks import sk_unit02 as sk_unit
from my_blocks import sk_unit02, sk_unit03
from keras.layers import Concatenate, Conv2DTranspose, concatenate, Dropout
from keras.regularizers import l2
import numpy as np

from collections import OrderedDict


import keras
from distutils.version import StrictVersion

if StrictVersion(keras.__version__) < StrictVersion('2.2.0'):
    from keras.applications.imagenet_utils import _obtain_input_shape
else:
    from keras_applications.imagenet_utils import _obtain_input_shape

from my_params import get_conv_params
from my_params import get_bn_params

dropout_rate = 0.5
act = "relu"


def coarse(feature, v, h, w, c):
    f = np.zeros((h, w, c))
    for i in range(c):
        idx = np.argmax(v)
        f[:, :, i] = feature[:, :, idx]
        v[idx] = -1
    return f


def fine(fa, fb, h, w, c):
    f = np.zeros((h, w, c))
    m = np.zeros((c, c))
    l = []
    for i in range(c):
        for j in range(c):
            if j not in l:
                a_one = (fa[:, :, i] - np.mean(fa[:, :, i]))/np.std(fa[:, :, i])
                b_one = (fb[:, :, j] - np.mean(fb[:, :, j]))/np.std(fb[:, :, j])
                m[i, j] = np.average(a_one*b_one)
        idx = np.argmax(m[i, :])
        l.append(idx)
        f[:, :, i] = fb[:, :, idx]
    return f


def build_mycov(
        repetitions=(2, 2, 2, 2),
        input_tensor=None,
        input_shape=None,
        classes=1,
        first_filters=64):

    """
    TODO
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    conv_params = get_conv_params(padding='same')
    init_filters = first_filters

    # resnext bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)

    # down
    dw_h_convs = OrderedDict()
    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            filters = init_filters * (2 ** stage)
            x = SK_change_block(filters, stage, block, behind_sknet=True)(x)
            if block == 1:
                dw_h_convs[stage] = x
                x = MaxPooling2D((2, 2), name='pooling'+str(stage+1))(x)

    filters = init_filters * (2 ** 4)
    x = SK_change_block(filters, 4, 0, behind_sknet=True)(x)
    x = SK_change_block(filters, 4, 1, behind_sknet=True)(x)
    # up
    for stage, rep in enumerate(repetitions):
        stage += 5
        filters = init_filters * (2 ** (4 - stage + 5 - 1))
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), name='up' + str(stage + 1), padding='same')(x)
        x = Concatenate(axis=-1, name='merge'+str(stage+1)+str(9 - stage))([x, dw_h_convs[8 - stage]])

        for block in range(rep):
            x = SK_change_block(filters, stage, block, behind_sknet=False)(x)

    x = Conv2D(classes, (1, 1), activation='sigmoid', name='output', **conv_params)(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = Model(inputs, x)

    return model


def build_mycov11(
        repetitions=(2, 2, 2, 2),
        input_tensor=None,
        input_shape=None,
        classes=1,
        first_filters=64):

    """
    TODO
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    conv_params = get_conv_params(padding='same')
    init_filters = first_filters

    # resnext bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)

    # down
    dw_h_convs = OrderedDict()
    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            filters = init_filters * (2 ** stage)
            x = sk_unit02(filters, stage, block)(x)
            if block == 1:
                dw_h_convs[stage] = x
                x = MaxPooling2D((2, 2), name='pooling'+str(stage+1))(x)

    filters = init_filters * (2 ** 4)
    x = sk_unit02(filters, 4, 0)(x)
    x = sk_unit02(filters, 4, 1)(x)
    # up
    for stage, rep in enumerate(repetitions):
        stage += 5
        filters = init_filters * (2 ** (4 - stage + 5 - 1))
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), name='up' + str(stage + 1), padding='same')(x)
        x = Concatenate(axis=-1, name='merge'+str(stage+1)+str(9 - stage))([x, dw_h_convs[8 - stage]])

        for block in range(rep):
            x = sk_unit02(filters, stage, block)(x)

    x = Conv2D(classes, (1, 1), activation='sigmoid', name='output', **conv_params)(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = Model(inputs, x)

    return model


def build_mycov12(
        repetitions=(2, 2, 2, 2),
        input_tensor=None,
        input_shape=None,
        classes=1,
        first_filters=64):

    """
    TODO
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    conv_params = get_conv_params(padding='same')
    init_filters = first_filters

    # resnext bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)

    # down
    dw_h_convs = OrderedDict()
    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            filters = init_filters * (2 ** stage)
            x = sk_unit03(filters, stage, block)(x)
            if block == 1:
                dw_h_convs[stage] = x
                x = MaxPooling2D((2, 2), name='pooling'+str(stage+1))(x)

    filters = init_filters * (2 ** 4)
    x = sk_unit03(filters, 4, 0)(x)
    x = sk_unit03(filters, 4, 1)(x)
    # up
    for stage, rep in enumerate(repetitions):
        stage += 5
        filters = init_filters * (2 ** (4 - stage + 5 - 1))
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), name='up' + str(stage + 1), padding='same')(x)
        x = Concatenate(axis=-1, name='merge'+str(stage+1)+str(9 - stage))([x, dw_h_convs[8 - stage]])

        for block in range(rep):
            x = sk_unit03(filters, stage, block)(x)

    x = Conv2D(classes, (1, 1), activation='sigmoid', name='output', **conv_params)(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = Model(inputs, x)

    return model


def standard_unit(input_tensor, stage, nb_filter):
    my_stage = int(stage)
    x = sk_unit(nb_filter, my_stage-1, 0, r=16, L=32)(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = sk_unit(nb_filter, my_stage-1, 1, r=16, L=32)(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x


def build_myconv01(img_rows, img_cols, color_type=1, num_class=1, filter_rate=1):

    if filter_rate == 1:
        nb_filter = [32, 64, 128, 256, 512]
    if filter_rate == 2:
        nb_filter = [64, 128, 256, 512, 1024]
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    model = Model(img_input, unet_output)

    return model


def standard_unit_deepwise(input_tensor, stage, nb_filter):
    # deepwiseconv
    my_stage = int(stage)
    x = SK_block_deepwise(nb_filter, my_stage-1, 0, r=16, L=32)(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = SK_block_deepwise(nb_filter, my_stage-1, 1, r=16, L=32)(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x

def build_myconv03(img_rows, img_cols, color_type=1, num_class=1, filter_rate=1):

    if filter_rate == 1:
        nb_filter = [32, 64, 128, 256, 512]
    if filter_rate == 2:
        nb_filter = [64, 128, 256, 512, 1024]
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit_deepwise(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit_deepwise(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit_deepwise(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit_deepwise(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit_deepwise(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit_deepwise(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit_deepwise(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit_deepwise(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit_deepwise(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    model = Model(img_input, unet_output)

    return model
# model = build_myconv01(512, 512, 1)
# model.summary()