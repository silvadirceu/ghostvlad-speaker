import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras import activations


import numpy as np
weight_decay = 1e-4

def identity_block_2D(x, kernel_size, filters, stage, block, trainable=True):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    x_skip = x # this will be used for addition with the residual block
    f1, f2, f3 = filters
    bn_axis = 3

    #first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
             trainable=trainable, use_bias=False,
             kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
             kernel_initializer=tf.keras.initializers.Orthogonal())(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable)(x)
    x = Activation(activations.relu)(x)

    #second block # bottleneck (but size kept same with padding)
    x = Conv2D(f2, kernel_size=[kernel_size, kernel_size], strides=(1, 1), padding='same',
             trainable=trainable, use_bias=False,
             kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
             kernel_initializer=tf.keras.initializers.Orthogonal())(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable)(x)

    x = Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = Conv2D(f3, kernel_size=[1, 1], strides=(1, 1), padding='valid',
             trainable=trainable, use_bias=False,
             kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
             kernel_initializer=tf.keras.initializers.Orthogonal())(x)

    x = BatchNormalization(axis=bn_axis, trainable=trainable)(x)

    # add the input
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def conv_block_2D(x, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """

    x_skip = x # this will be used for addition with the residual block
    f1, f2, f3 = filters
    bn_axis = 3

    #first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=strides, padding='valid',
             trainable=trainable,  use_bias=False,
             kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
             kernel_initializer=tf.keras.initializers.Orthogonal())(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable)(x)
    x = Activation(activations.relu)(x)

    #second block # bottleneck (but size kept same with padding)
    x = Conv2D(f2, kernel_size=[kernel_size, kernel_size], strides=(1, 1), padding='same',
             trainable=trainable, use_bias=False,
             kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
             kernel_initializer=tf.keras.initializers.Orthogonal())(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable)(x)
    x = Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = Conv2D(f3, kernel_size=[1, 1], strides=(1, 1), padding='valid',
             trainable=trainable, use_bias=False,
             kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
             kernel_initializer=tf.keras.initializers.Orthogonal())(x)

    x = BatchNormalization(axis=bn_axis, trainable=trainable)(x)

    # shortcut
    x_skip = Conv2D(f3, kernel_size=[1, 1], strides=strides, padding='valid',
             trainable=trainable, use_bias=False,
             kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
             kernel_initializer=tf.keras.initializers.Orthogonal())(x_skip)

    x_skip = BatchNormalization(axis=bn_axis, trainable=trainable)(x_skip)

    # add
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def model_resnet_2D_v2(input_shape, trainable=True):
    bn_axis = 3

    # ===============================================
    #            Input Block
    # ===============================================
    input_ = Input(shape=input_shape) #inputs.shape[1:])

    # ===============================================
    #            Convolution Block 1
    # ===============================================

    x1 = Conv2D(64, [7, 7],
                kernel_initializer=tf.keras.initializers.Orthogonal(),
                use_bias=False, trainable=trainable,
                kernel_regularizer=tf.keras.regularizers.L2(l2=weight_decay),
                padding='same')(input_)
    x1 = BatchNormalization(axis=bn_axis, trainable=trainable)(x1)
    x1 = Activation(activations.relu)(x1)
    x1 = MaxPooling2D([2, 2], strides=[2, 2])(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(x1, 3, [48, 48, 96], stage=2, block='a', strides=(1, 1), trainable=trainable)
    x2 = identity_block_2D(x2, 3, [48, 48, 96], stage=2, block='b', trainable=trainable)

    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(x2, 3, [96, 96, 128], stage=3, block='a', trainable=trainable)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='b', trainable=trainable)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='c', trainable=trainable)
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(x3, 3, [128, 128, 256], stage=4, block='a', trainable=trainable)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='b', trainable=trainable)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='c', trainable=trainable)
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(x4, 3, [256, 256, 512], stage=5, block='a', trainable=trainable)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='b', trainable=trainable)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='c', trainable=trainable)

    y = MaxPooling2D([3, 1], strides=[2, 1])(x5)

    model = Model(inputs=[input_], outputs=[y])

    return model


if __name__ == '__main__':
    # assignValue()

    import preprocess
    params = {'dim': (257, None, 1),
            'nfft': 512,
            'min_slice': 720,
            'win_length': 400,
            'hop_length': 160,
            'n_classes': 5994,
            'sampling_rate': 16000,
            'normalize': True,
            }
    specs = preprocess.load_data(r'D:\PythonSpace\Speaker-Diarization\ghostvlad\4persons\a_1.wav', split=False, win_length=params['win_length'], sr=params['sampling_rate'],
                       hop_length=params['hop_length'], n_fft=params['nfft'],
                       min_slice=params['min_slice'])
    specs = np.expand_dims(np.expand_dims(specs[0], 0), -1)

    with tf.Session() as sess:
        inputs, y = resnet_2D_v2([1, 257, None, 1], mode='eval')
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(sess, "ckpt/model.ckpt")

        output = tf.get_default_graph().get_tensor_by_name("mpool2/max_pooling2d/MaxPool:0")
        output = sess.run(output, feed_dict={"input:0": specs})
        print(output)




