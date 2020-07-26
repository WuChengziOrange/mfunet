from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Add
from keras.layers import Lambda
from keras.layers import Concatenate
from keras.layers import ZeroPadding2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense, DepthwiseConv2D, Softmax, Reshape
import tensorflow as tf

from my_params import get_conv_params
from my_params import get_bn_params


def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    add_name = name_base + 'add'
    attention_name = name_base + 'attention_'
    return conv_name, bn_name, relu_name, sc_name, add_name, attention_name


def GroupConv2D(filters, kernel_size, conv_params, conv_name, strides=(1, 1), cardinality=32):
    """
    :param filters:  useless param,existing for harmonious
    :param kernel_size: size of kernel
    :param conv_params:
    :param conv_name:
    :param strides:tuple of integers, strides for conv (3x3) layer in block
    :param cardinality: 32,if it > channels,set to 1
    :return:
    """
    def layer(input_tensor, cardinality=cardinality):

        if input_tensor.shape[-1] < cardinality:
            cardinality = 1

        grouped_channels = int(input_tensor.shape[-1]) // cardinality

        blocks = []
        for c in range(cardinality):
            x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input_tensor)
            name = conv_name + '_' + str(c)
            x = Conv2D(grouped_channels, kernel_size, strides=strides,
                       name=name, **conv_params)(x)
            blocks.append(x)

        if cardinality == 1:
            return blocks[0]

        x = Concatenate(axis=-1)(blocks)
        return x
    return layer


def scale_feature(L):
    scale = L[0]
    feature_map = L[1]
    x = tf.multiply(scale, feature_map)
    return x


def SK_block(filters, stage, block, r=16, L=32):

    def layer(input_tensor):
        # extracting params and names for layers
        conv_params = get_conv_params(padding='same')
        conv_params_dilation = get_conv_params(padding='same', dilation_rate=2)
        bn_params = get_bn_params()
        fc_bn_params = get_bn_params(axis=-1)
        conv_name, bn_name, relu_name, sc_name, add_name, attention_name = handle_block_names(stage, block)

        x_3 = GroupConv2D(filters, (3, 3), conv_params, conv_name + '3_3')(input_tensor)
        x_3 = BatchNormalization(name=bn_name + '3_3', **bn_params)(x_3)
        x_3 = Activation('relu', name=relu_name + '3_3')(x_3)

        x_5 = GroupConv2D(filters, (3, 3), conv_params_dilation, conv_name + '5_5')(input_tensor)
        x_5 = BatchNormalization(name=bn_name + '5_5', **bn_params)(x_5)
        x_5 = Activation('relu', name=relu_name + '5_5')(x_5)
        u = Add(name=add_name)([x_3, x_5])

        s = GlobalAveragePooling2D(name=attention_name+'gap')(u)
        d = max(int(s.shape[-1])//r, L)
        z = Dense(d, name=attention_name + 'fc1')(s)
        z = BatchNormalization(name=bn_name + 'vector', **fc_bn_params)(z)
        z = Activation('relu', name=relu_name + 'vector')(z)
        c = s.get_shape().as_list()[-1]
        z1 = Dense(c*2, name=attention_name + 'fc2')(z)
        z1 = Reshape([2, -1], name=attention_name + 'reshape1')(z1)
        soft = Softmax(axis=1, name=attention_name+'softmax')(z1)
        A = Lambda(lambda vector: vector[:, 0, :])(soft)
        A = Reshape([1, 1, -1], name=attention_name + 'reshape_A')(A)
        B = Lambda(lambda vector: vector[:, 1, :])(soft)
        B = Reshape([1, 1, -1], name=attention_name + 'reshape_B')(B)
        x_3 = Lambda(scale_feature)([A, x_3])
        x_5 = Lambda(scale_feature)([B, x_5])
        v = Add()([x_3, x_5])
        return v
    return layer


def SK_change_block(filters, stage, block, behind_sknet=True, r=16, L=32):
    def layer(x):
        # extracting params and names for layers
        conv_params = get_conv_params(padding='same')
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name, add_name, attention_name = handle_block_names(stage, block)

        if not behind_sknet:
            x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)
            x = BatchNormalization(name=bn_name + '1x1_1', **bn_params)(x)
            x = Activation('relu', name=relu_name + '1x1_1')(x)

        x = SK_block(filters, stage, block, r=r, L=L)(x)

        if behind_sknet:
            x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)
            x = BatchNormalization(name=bn_name + '1x1_1', **bn_params)(x)
            x = Activation('relu', name=relu_name + '1x1_1')(x)
        return x
    return layer


def sk_unit(filters, stage, block, r=16, L=32):
    def layer(x):
        # extracting params and names for layers
        conv_params = get_conv_params(padding='same')
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name, add_name, attention_name = handle_block_names(stage, block)

        x = Conv2D(int(filters)//2, (1, 1), name=conv_name + '1', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '1x1_1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1x1_1')(x)

        x = SK_block(filters, stage, block, r=r, L=L)(x)

        x = Conv2D(filters, (1, 1), name=conv_name + '2', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '1x1_2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1x1_2')(x)
        return x
    return layer


def sk_unit01(filters, stage, block, r=16, L=32):
    def layer(x):
        # extracting params and names for layers
        conv_params = get_conv_params(padding='same')
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name, add_name, attention_name = handle_block_names(stage, block)
        if x.shape[-1] < filters:
            behind_sknet = True
        else:
            behind_sknet = False

        if not behind_sknet:
            x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)
            x = BatchNormalization(name=bn_name + '1x1_1', **bn_params)(x)
            x = Activation('relu', name=relu_name + '1x1_1')(x)

        x = SK_block(filters, stage, block, r=r, L=L)(x)

        if behind_sknet:
            x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)
            x = BatchNormalization(name=bn_name + '1x1_1', **bn_params)(x)
            x = Activation('relu', name=relu_name + '1x1_1')(x)
        return x
    return layer

def sk_unit02(filters, stage, block, r=16, L=32):
    def layer(input_tensor):
        # extracting params and names for layers
        conv_params = get_conv_params(padding='same')
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name, add_name, attention_name = handle_block_names(stage, block)

        x = Conv2D(int(filters)//2, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = BatchNormalization(name=bn_name + '1x1_1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1x1_1')(x)

        x = SK_block(filters, stage, block, r=r, L=L)(x)

        x = Conv2D(filters, (1, 1), name=conv_name + '2', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '1x1_2', **bn_params)(x)

        shortcut = Conv2D(filters, (1, 1), name=sc_name, **conv_params)(input_tensor)
        shortcut = BatchNormalization(name=sc_name + '_bn', **bn_params)(shortcut)
        x = Add()([x, shortcut])

        x = Activation('relu', name=relu_name + 'addout')(x)
        return x
    return layer


def sk_unit03(filters, stage, block, r=16, L=32):
    def layer(input_tensor):
        # extracting params and names for layers
        conv_params = get_conv_params(padding='same')
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name, add_name, attention_name = handle_block_names(stage, block)

        x = Conv2D(int(filters)//2, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = BatchNormalization(name=bn_name + '1x1_1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1x1_1')(x)

        x = SK_block_deepwise(filters, stage, block, r=r, L=L)(x)

        x = Conv2D(filters, (1, 1), name=conv_name + '2', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '1x1_2', **bn_params)(x)

        shortcut = Conv2D(filters, (1, 1), name=sc_name, **conv_params)(input_tensor)
        shortcut = BatchNormalization(name=sc_name + '_bn', **bn_params)(shortcut)
        x = Add()([x, shortcut])

        x = Activation('relu', name=relu_name + 'addout')(x)
        return x
    return layer


def SK_block_deepwise(filters, stage, block, r=16, L=32):

    def layer(input_tensor):
        # extracting params and names for layers
        conv_params = get_conv_params(padding='same', use_bias=False)
        conv_params_dilation = get_conv_params(padding='same', use_bias=False, dilation_rate=2)
        bn_params = get_bn_params()
        fc_bn_params = get_bn_params(axis=-1)
        conv_name, bn_name, relu_name, sc_name, add_name, attention_name = handle_block_names(stage, block)

        x_3 = DepthwiseConv2D((3, 3), padding='same', name=conv_name + '3_3')(input_tensor)
        x_3 = BatchNormalization(name=bn_name + '3_3', **bn_params)(x_3)
        x_3 = Activation('relu', name=relu_name + '3_3')(x_3)

        x_5 = DepthwiseConv2D((5, 5), padding='same', name=conv_name + '5_5')(input_tensor)
        x_5 = BatchNormalization(name=bn_name + '5_5', **bn_params)(x_5)
        x_5 = Activation('relu', name=relu_name + '5_5')(x_5)
        u = Add(name=add_name)([x_3, x_5])

        s = GlobalAveragePooling2D(name=attention_name+'gap')(u)
        d = max(int(s.shape[-1])//r, L)
        z = Dense(d, name=attention_name + 'fc1')(s)
        z = BatchNormalization(name=bn_name + 'vector', **fc_bn_params)(z)
        z = Activation('relu', name=relu_name + 'vector')(z)
        c = s.get_shape().as_list()[-1]
        z1 = Dense(c*2, name=attention_name + 'fc2')(z)
        z1 = Reshape([2, -1], name=attention_name + 'reshape1')(z1)
        soft = Softmax(axis=1, name=attention_name+'softmax')(z1)
        A = Lambda(lambda vector: vector[:, 0, :])(soft)
        A = Reshape([1, 1, -1], name=attention_name + 'reshape_A')(A)
        B = Lambda(lambda vector: vector[:, 1, :])(soft)
        B = Reshape([1, 1, -1], name=attention_name + 'reshape_B')(B)
        x_3 = Lambda(scale_feature)([A, x_3])
        x_5 = Lambda(scale_feature)([B, x_5])
        v = Add()([x_3, x_5])
        return v
    return layer


def conv_block(filters, stage, block, strides=(2, 2)):
    """The conv block is the block that has conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        strides: tuple of integers, strides for conv (3x3) layer in block
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):

        # extracting params and names for layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), conv_params, conv_name + '2', strides=strides)(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)

        x = Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)

        shortcut = Conv2D(filters*2, (1, 1), name=sc_name, strides=strides, **conv_params)(input_tensor)
        shortcut = BatchNormalization(name=sc_name+'_bn', **bn_params)(shortcut)
        x = Add()([x, shortcut])

        x = Activation('relu', name=relu_name)(x)
        return x

    return layer


def identity_block(filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = Activation('relu', name=relu_name + '1')(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), conv_params, conv_name + '2')(x)
        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)

        x = Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)

        x = Add()([x, input_tensor])

        x = Activation('relu', name=relu_name)(x)
        return x

    return layer
