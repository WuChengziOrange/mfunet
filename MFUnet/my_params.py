# default parameters for convolution and batchnorm layers of ResNet models
# parameters are obtained from MXNet converted model
from keras.regularizers import l2


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_normal',
        'use_bias': True,
        'padding': 'same',
        'dilation_rate': 1,
        'kernel_regularizer': l2(1e-4)
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params
