from tensorflow import pad
from tensorflow.keras.layers import (
    BatchNormalization, ReLU, Conv2D, Add, GlobalAveragePooling2D, Dense, Lambda, MaxPooling2D)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import int_shape

def bn_act_block(inputs, norm_momentum) :
    '''
        BatchNormalization은 Conv2D와 activation 사이에 위치함
    '''
    return ReLU()(
        BatchNormalization(momentum = norm_momentum)(inputs)
    )

def _Conv2D(inputs, channels, size, strides, init, reg) :
    return Conv2D(filters = channels, k_size = (size, size), strides = (strides, strides), padding = 'same',
        activation = None, use_bias = False, kernel_initializer = init, kernel_regularizer = reg)(inputs)

def shortcut_zero_pad(inputs, channels, strides) :
    return Lambda(
        lambda x : pad(x, [[0, 0], [0, 0], [0, 0], [0, channels]]))(
            MaxPooling2D(pool_size = (1, 1), strides = (strides, strides), padding = 'same')(inputs))

def bottleneck_block(inputs, out_channels, size, strides, init, reg, norm_momentum, pre_activation, share_first_outputs = False) :
    '''
        share_first_outputs : pre-activation model에서 1번째 layer 또는 resizing을 할 때
        shortcut_inputs = ReLU()(
            BatchNormalization()(F(x) + x)
        )
        share_first_outputs는 pre_activation일 때만 적용됨
    '''
    if pre_activation :     # BatchNormalization -> Activation -> Conv2D
        pre_activation_outputs = bn_act_block(inputs, norm_momentum)
        if share_first_outputs :
            shortcut_inputs = pre_activation_outputs

        outputs = _Conv2D(pre_activation_outputs, out_channels // 4, 1, 1, init, reg)

        outputs = _Conv2D(
            bn_act_block(outputs, norm_momentum), out_channels // 4, size, strides, init, reg
        )
        outputs = _Conv2D(
            bn_act_block(outputs, norm_momentum), out_channels, 1, 1, init, reg
        )
    else :                  # Conv2D -> BatchNormalization -> Activation
        outputs = bn_act_block(
            _Conv2D(inputs, out_channels // 4, 1, 1, init, reg), norm_momentum
        )
        outputs = bn_act_block(
            _Conv2D(outputs, out_channels // 4, size, strides, init, reg), norm_momentum
        )
        outputs = BatchNormalization(momentum = norm_momentum)(
            _Conv2D(outputs, out_channels, 1, 1, init, reg)
        )

    try :
        return [outputs, shortcut_inputs]
    except NameError :
        return [outputs]

def non_bottleneck_block(inputs, out_channels, size, strides, init, reg, norm_momentum, pre_activation, share_first_outputs = False) :
    '''
        share_first_outputs : pre-activation model에서 1번째 layer 또는 resizing을 할 때
        shortcut_inputs = ReLU()(
            BatchNormalization()(F(x) + x)
        )
    '''
    if pre_activation :
        pre_activation_outputs = bn_act_block(inputs, norm_momentum)
        if share_first_outputs :
            shortcut_inputs = pre_activation_outputs

        outputs = _Conv2D(pre_activation_outputs, out_channels, size, strides, init, reg)

        outputs = _Conv2D(
            bn_act_block(outputs, norm_momentum), out_channels, size, 1, init, reg
        )
    else :
        outputs = bn_act_block(
            _Conv2D(inputs, out_channels, size, strides, init, reg), norm_momentum
        )
        outputs = BatchNormalization(momentum = norm_momentum)(
            _Conv2D(outputs, out_channels, size, 1, init, reg)
        )

    try :
        return [outputs, shortcut_inputs]
    except NameError :
        return [outputs]

def make_residual_block(inputs, in_channels, out_channels, size, strides, init, reg, norm_momentum,
    pre_activation, bottleneck, zero_pad, share_first_outputs = False) :

    '''
        non-linear path : F(x)
        shortcut에 MaxPooling2D를 적용할 때는 pre-activation을 적용하지 않음
    '''
    if bottleneck :
        func_outputs = bottleneck_block(
            inputs, out_channels, size, strides, init, reg, norm_momentum, pre_activation, share_first_outputs
        )

    else :
        func_outputs = non_bottleneck_block(
            inputs, out_channels, size, strides, init, reg, norm_momentum, pre_activation, share_first_outputs
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
        if zero_pad :
            shortcut_outputs = shortcut_zero_pad(shortcut_inputs, out_channels - in_channels, strides)
        else :
            if pre_activation :
                shortcut_outputs = _Conv2D(shortcut_inputs, out_channels, 1, strides, init, reg)
            else :
                shortcut_outputs = BatchNormalization(momentum = norm_momentum)(
                    _Conv2D(shortcut_inputs, out_channels, 1, strides, init, reg)
                )

        residual_block_outputs = Add()([non_linear_outputs, shortcut_outputs])
    else :
        residual_block_outputs = Add()([non_linear_outputs, shortcut_inputs])

    if not pre_activation : # G(F(x) + x)
        residual_block_outputs = ReLU()(residual_block_outputs)

    return residual_block_outputs

def make_resnet_model(inputs, *args, **kwargs) :
    num_filters = kwargs["num_filters"]
    k_size = kwargs["kernel_size"]
    k_init = kwargs['kernel_initializer']
    l2_value = kwargs["l2_value"]
    norm_momentum = kwargs["norm_momentum"]
    num_box = kwargs["num_box"]
    num_blocks_in_box = kwargs["num_blocks_in_box"]
    num_categories = kwargs["num_categories"]
    bottleneck = kwargs["bottleneck"]
    zero_pad = kwargs["zero_pad"]
    pre_activation = kwargs["pre_activation"]

    # image shape : [batch_size, height, width, num_channels]
    assert num_box > 0 and num_box == len(num_blocks_in_box)
    assert (num_box == len(num_filters)) or (num_box + 1 == len(num_filters))
    
    first_conv_outputs = _Conv2D(inputs, num_filters[0], k_size, 1, k_init, l2(l2_value))
    inputs = first_conv_outputs
    in_channels = int_shape(inputs)[-1]

    status = (num_box == len(num_filters))
    if status :
        for_range = range(len(num_filters))
        base_idx = 0
    else :
        for_range = range(1, len(num_filters))
        base_idx = 1
        
    if pre_activation :
        for outer_idx in for_range :
            if status :
                num_blocks = num_blocks_in_box[outer_idx]
            else :
                num_blocks = num_blocks_in_box[outer_idx - 1]

            for inner_idx in range(num_blocks) :
                if (outer_idx != base_idx) and (inner_idx == 0) :
                    outputs = make_residual_block(inputs, in_channels, num_filters[outer_idx], k_size,
                        2, k_init, l2(l2_value), norm_momentum, True, bottleneck, zero_pad,
                        share_first_outputs = True if not zero_pad else False)
                    
                    inputs = outputs
                    in_channels = int_shape(inputs)[-1]

                else :
                    outputs = make_residual_block(inputs, in_channels, num_filters[outer_idx], k_size,
                        1, k_init, l2(l2_value), norm_momentum, True, bottleneck, zero_pad, inner_idx == 0)

                    inputs = outputs
                    in_channels = int_shape(inputs)[-1]

        outputs = bn_act_block(inputs, norm_momentum)

    else :
        outputs = bn_act_block(inputs, norm_momentum)
        inputs = outputs

        for outer_idx in for_range :
            if status :
                num_blocks = num_blocks_in_box[outer_idx]
            else :
                num_blocks = num_blocks_in_box[outer_idx - 1]

            for inner_idx in range(num_blocks) :
                if (outer_idx != base_idx) and (inner_idx == 0) :
                    outputs = make_residual_block(inputs, in_channels, num_filters[outer_idx], k_size,
                        2, k_init, l2(l2_value), norm_momentum, False, bottleneck, zero_pad)

                    inputs = outputs
                    in_channels = int_shape(inputs)[-1]

                else :
                    outputs = make_residual_block(inputs, in_channels, num_filters[outer_idx], k_size,
                        1, k_init, l2(l2_value), norm_momentum, False, bottleneck, zero_pad)

                    inputs = outputs
                    in_channels = int_shape(inputs)[-1]

    avgPool_outputs = GlobalAveragePooling2D()(outputs)
    logits = Dense(units = num_categories, activation = 'softmax', 
        k_initializer = k_init, kernel_regularizer= l2(l2_value))(avgPool_outputs)

    return logits