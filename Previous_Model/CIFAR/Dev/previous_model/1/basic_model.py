from tensorflow import pad
from tensorflow.keras.layers import (
    BatchNormalization, ReLU, Conv2D, Add, GlobalAveragePooling2D, Dense, Lambda, MaxPooling2D)
from tensorflow.keras.regularizers import l2

def make_layers(inputs, size, l2_value, norm_momentum, kernel_init, in_channels, out_channels,
    bottleneck = False, zero_pad = True) :
    status = (in_channels != out_channels)
    shortcut_inputs = inputs

    if not bottleneck :
        # if num_filters = 256, filter_size = 32

        # num_filters = 256 or 512, filter_size = 32 or 16
        first_conv_outputs = Conv2D(filters = out_channels, kernel_size = [size, size],
            strides = (2, 2) if status else (1, 1), padding = 'same', activation = None,
            kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(inputs)
        first_norm_outputs = BatchNormalization(momentum = norm_momentum)(first_conv_outputs)
        first_act_outputs = ReLU()(first_norm_outputs)

        # num_filters = 256 or 512, filter_size = 32 or 16
        second_conv_outputs = Conv2D(filters = out_channels, kernel_size = [size, size],
            padding = 'same', activation = None, 
            kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(first_act_outputs)
        non_linear_outputs = BatchNormalization(momentum = norm_momentum)(second_conv_outputs)

    else :
        # if num_filters = 256, filter_size = 32

        # num_filters = 64, filter_size = 32
        first_conv_outputs = Conv2D(filters = in_channels // 4, kernel_size = [1, 1], strides = (1, 1), 
            padding = 'same', activation = None, 
            kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(inputs)
        first_norm_outputs = BatchNormalization(momentum = norm_momentum)(first_conv_outputs)
        first_act_outputs = ReLU()(first_norm_outputs)

        # num_filters = 64, filter_size = 32 or 16
        second_conv_outputs = Conv2D(filters = in_channels // 4, kernel_size = [size, size], 
            strides = (2, 2) if status else (1, 1), padding = 'same', activation = None, 
            kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(first_act_outputs)
        second_norm_outputs = BatchNormalization(momentum = norm_momentum)(second_conv_outputs)
        second_act_outputs = ReLU()(second_norm_outputs)

        # num_filters = 256 or 512, filter_size = 32 or 16
        third_conv_outputs = Conv2D(filters = out_channels, kernel_size = [1, 1], strides =(1, 1),
            padding = 'same', activation = None, kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(second_act_outputs)
        non_linear_outputs = BatchNormalization(momentum = norm_momentum)(third_conv_outputs)

    if status :
        if zero_pad :
            shortcut_proj = MaxPooling2D(pool_size = (1, 1), strides = (2, 2), padding = 'same')(shortcut_inputs)

            # shortcuts_proj_outputs = [batch_size, image_height, image_width, in_channels]
            # num_channels 부분만 뒤 쪽에 zero padding을 함.
            shortcut_outputs = Lambda(lambda x : pad(x, [[0, 0], [0, 0], [0, 0], [0, out_channels - in_channels]]))(shortcut_proj)
        else :
            shortcut_proj = Conv2D(filters = out_channels, kernel_size = [1, 1], strides = (2, 2), padding = 'same', 
                activation = None, use_bias = False, 
                kernel_initializer = kernel_init, kernel_regularizer= l2(l2_value))(shortcut_inputs)
            shortcut_outputs = BatchNormalization(momentum = norm_momentum)(shortcut_proj)

        return Add()([non_linear_outputs, shortcut_outputs])
    else :
        return Add()([non_linear_outputs, shortcut_inputs])

def make_basic_model(inputs, *args, **kwargs) :
    num_filters = kwargs["num_filters"]
    kernel_size = kwargs["kernel_size"]
    kernel_init = kwargs['kernel_initializer']
    l2_value = kwargs["l2_value"]
    norm_momentum = kwargs["norm_momentum"]
    num_box = kwargs["num_box"]
    num_blocks_in_box = kwargs["num_blocks_in_box"]
    num_categories = kwargs["num_categories"]
    bottleneck = kwargs["bottleneck"]
    zero_pad = kwargs["zero_pad"]

    # image shape : [batch_size, height, width, num_channels]
    assert num_box > 0 and num_box == len(num_blocks_in_box) == len(num_filters)

    first_conv_outputs = Conv2D(filters = num_filters[0], kernel_size = [kernel_size, kernel_size],
        strides = (1, 1), padding = 'same', activation = None, use_bias = False,
        kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(inputs)
    first_norm_outputs = BatchNormalization(momentum = norm_momentum)(first_conv_outputs)
    first_act_outputs = ReLU()(first_norm_outputs)
    
    inputs = first_act_outputs
    for idx in range(num_box) :
        for _ in range(num_blocks_in_box[idx]) :
            outputs = make_layers(inputs, kernel_size, l2_value, norm_momentum, kernel_init,
                num_filters[idx], num_filters[idx], bottleneck = bottleneck, zero_pad = zero_pad)
            inputs = outputs
        
        if idx != num_box - 1 : # 마지막 block은 size를 줄이지 않음
            outputs = make_layers(inputs, kernel_size, l2_value, norm_momentum, kernel_init, 
                num_filters[idx], num_filters[idx + 1], bottleneck = bottleneck, zero_pad = zero_pad)
            inputs = outputs            
        
    final_outputs = inputs
    avgPool_outputs = GlobalAveragePooling2D()(final_outputs)
    logits = Dense(units = num_categories, activation = 'softmax', 
        kernel_initializer = kernel_init, kernel_regularizer= l2(l2_value))(avgPool_outputs)

    return logits