from tensorflow import pad
from tensorflow.keras.layers import (
    BatchNormalization, ReLU, Conv2D, Add, GlobalAveragePooling2D, Dense, Lambda, MaxPooling2D)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import int_shape

def bn_act_block(inputs, momentum) :
    '''
        BatchNormalization은 Conv2D와 activation 사이에 위치함
    '''
    return ReLU()(
        BatchNormalization(momentum = momentum)(inputs)
    )

def _Conv2D(inputs, channels, size, strides, init, reg) :
    return Conv2D(filters = channels, kernel_size = (size, size), strides = (strides, strides), padding = 'same',
        activation = None, use_bias = False, kernel_initializer = init, kernel_regularizer = reg)(inputs)

def shortcut_zero_pad(inputs, channels, strides) :
    return Lambda(
        lambda x : pad(x, [[0, 0], [0, 0], [0, 0], [0, channels]]))(
            MaxPooling2D(pool_size = (1, 1), strides = (strides, strides), padding = 'same')(inputs))

def non_bottleneck_block(inputs, out_channels, size, strides, init, reg, momentum) :
    outputs = bn_act_block(
        _Conv2D(inputs, out_channels, size, strides, init, reg), momentum
    )
    outputs = BatchNormalization(momentum = momentum)(
        _Conv2D(outputs, out_channels, size, 1, init, reg)
    )

    return outputs

def make_residual_block(inputs, in_channels, out_channels, size, strides, init, reg, momentum,
    zero_pad) :

    # on-linear path : F(x)
    shortcut_inputs = inputs
    non_linear_outputs = non_bottleneck_block(
        inputs, out_channels, size, strides, init, reg, momentum)

    if in_channels != out_channels :
        if zero_pad :
            shortcut_outputs = shortcut_zero_pad(shortcut_inputs, out_channels - in_channels, strides)
        else :
            shortcut_outputs = BatchNormalization(momentum = momentum)(
                _Conv2D(shortcut_inputs, out_channels, 1, strides, init, reg)
            )

        residual_block_outputs = Add()([non_linear_outputs, shortcut_outputs])
    else :
        residual_block_outputs = Add()([non_linear_outputs, shortcut_inputs])

    # G(F(x) + x)
    return ReLU()(residual_block_outputs)

def make_original_ResNet(inputs, **kwargs) :
    num_layers          = kwargs["num_layers"]
    k_init              = kwargs['kernel_initializer']
    momentum         = kwargs["momentum"]
    zero_pad            = kwargs["zero_pad"]

    num_blocks      = (num_layers - 2) // 6
    l2_value        = 0.0001
    num_filters     = [16, 32, 64]
        
    print("\nnum_blocks : ", num_blocks)

    # start
    first_conv_outputs  = _Conv2D(inputs, 16, 3, 1, k_init, l2(l2_value))
    outputs = bn_act_block(first_conv_outputs, momentum)
    inputs = outputs
    in_channels         = int_shape(inputs)[-1]

    for outer_idx in range(3) :
        for inner_idx in range(num_blocks) :
            if (outer_idx != 0) and (inner_idx == 0) :
                outputs = make_residual_block(inputs, in_channels, num_filters[outer_idx], 3,
                    2, k_init, l2(l2_value), momentum, zero_pad)

                inputs = outputs
                in_channels = int_shape(inputs)[-1]

            else :
                outputs = make_residual_block(inputs, in_channels, num_filters[outer_idx], 3,
                    1, k_init, l2(l2_value), momentum, zero_pad)

                inputs = outputs
                in_channels = int_shape(inputs)[-1]

    avgPool_outputs = GlobalAveragePooling2D()(outputs)
    logits = Dense(units = kwargs["num_categories"], activation = 'softmax', 
        kernel_initializer = k_init, kernel_regularizer= l2(l2_value))(avgPool_outputs)

    return logits