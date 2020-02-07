from tensorflow import pad
from tensorflow.keras.layers import (
    BatchNormalization, ReLU, Conv2D, Add, GlobalAveragePooling2D, Dense, Lambda, MaxPooling2D, Dropout)
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

def non_bottleneck_block(inputs, out_channels, size, strides, init, reg, momentum, 
    dropout_rate = 0, share_first_outputs = False) :
    '''
        share_first_outputs : pre-activation model에서 1번째 layer 또는 resizing을 할 때
        shortcut_inputs = ReLU()(
            BatchNormalization()(F(x) + x)
        )
    '''
    first_pre_activation_outputs = bn_act_block(inputs, momentum)
    if share_first_outputs :
        shortcut_inputs = first_pre_activation_outputs

    first_conv_outputs = _Conv2D(first_pre_activation_outputs, out_channels, size, strides, init, reg)

    second_pre_activation_outputs = bn_act_block(first_conv_outputs, momentum)
    if dropout_rate > 0 :
        dropout_outputs = Dropout(dropout_rate)(second_pre_activation_outputs)
        second_conv_outputs = _Conv2D(dropout_outputs, out_channels, size, 1, init, reg)
    else :
        second_conv_outputs = _Conv2D(second_pre_activation_outputs, out_channels, size, 1, init, reg)

    try :
        return [second_conv_outputs, shortcut_inputs]
    except NameError :
        return [second_conv_outputs]

def make_residual_block(inputs, in_channels, out_channels, size, strides, init, reg, momentum,
    dropout_rate = 0, share_first_outputs = False) :

    '''
        non-linear path : F(x)
        shortcut에 MaxPooling2D를 적용할 때는 pre-activation을 적용하지 않음
    '''

    func_outputs = non_bottleneck_block(
        inputs, out_channels, size, strides, init, reg, momentum, dropout_rate, share_first_outputs
    )

    if len(func_outputs) == 2 :
        non_linear_outputs = func_outputs[0]
        shortcut_inputs = func_outputs[1]
    else :
        non_linear_outputs = func_outputs[0]
        shortcut_inputs = inputs

    '''
        shortcut_inputs                 : x or ReLU(BN(x))
        pre-activation && resizing      => 첫 번째 ReLU(BN)을 x와 F(x)가 공유함
        not pre-activation && resizing  => ReLU(F(x) + BN(Conv2D(x)))
    '''
    if in_channels != out_channels :
        shortcut_outputs = _Conv2D(shortcut_inputs, out_channels, 1, strides, init, reg)

        residual_block_outputs = Add()([non_linear_outputs, shortcut_outputs])
    else :
        residual_block_outputs = Add()([non_linear_outputs, shortcut_inputs])

    return residual_block_outputs

def make_wide_ResNet(inputs, **kwargs) :
    num_layers          = kwargs["num_layers"]
    k_init              = kwargs['kernel_initializer']
    momentum         = kwargs["momentum"]

    num_blocks          = (kwargs["num_layers"] - 4) // (2 * 3)        
    l2_value            = 0.0005
    dropout_rate        = kwargs["dropout_rate"]
    num_filters         = [e * kwargs['kernel_width'] for e in [16, 32, 64]]

    print("num_blocks : ", num_blocks)

    # start
    first_conv_outputs  = _Conv2D(inputs, 16, 3, 1, k_init, l2(l2_value))
    inputs              = first_conv_outputs
    in_channels         = int_shape(inputs)[-1]

    for outer_idx in range(3) :
        for inner_idx in range(num_blocks) :
            if (outer_idx != 0) and (inner_idx == 0) :
                outputs = make_residual_block(inputs, in_channels, num_filters[outer_idx], 3,
                    2, k_init, l2(l2_value), momentum,
                    dropout_rate, True)
                    
                inputs = outputs
                in_channels = int_shape(inputs)[-1]

            else :
                outputs = make_residual_block(inputs, in_channels, num_filters[outer_idx], 3,
                    1, k_init, l2(l2_value), momentum, 
                    dropout_rate, inner_idx == 0)

                inputs = outputs
                in_channels = int_shape(inputs)[-1]

    outputs = bn_act_block(inputs, momentum)

    avgPool_outputs = GlobalAveragePooling2D()(outputs)
    logits = Dense(units = kwargs["num_categories"], activation = 'softmax', 
        kernel_initializer = k_init, kernel_regularizer= l2(l2_value))(avgPool_outputs)

    return logits