from tensorflow import pad
from tensorflow.keras.layers import (
    BatchNormalization, ReLU, Conv2D, Add, GlobalAveragePooling2D, Dense, Lambda, MaxPooling2D)
from tensorflow.keras.regularizers import l2

def make_layers(inputs, size, l2_value, norm_momentum, kernel_init, in_channels, out_channels, 
    first_layer = False, bottleneck = False, zero_pad = True) :

    # 1번째 layer 또는 dimension을 늘릴 때, shrotcuts에도 pre-activation 적용
    status = (in_channels != out_channels)
    if first_layer or status :
        norm_outputs = BatchNormalization(momentum = norm_momentum)(inputs)
        act_outputs = ReLU()(norm_outputs)
        shortcut_inputs = act_outputs
    else :
        shortcut_inputs = inputs
        norm_outputs = BatchNormalization(momentum = norm_momentum)(inputs)
        act_outputs = ReLU()(norm_outputs)

    if not bottleneck :
        first_conv_outputs = Conv2D(filters = out_channels, kernel_size = [size, size],
            strides = (2, 2) if status else (1, 1), padding = 'same', activation = None,
            kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(act_outputs)

        # num_filters = 256 or 512, filter_size = 32 or 16
        second_norm_outputs = BatchNormalization(momentum = norm_momentum)(first_conv_outputs)
        second_act_outputs = ReLU()(second_norm_outputs)
        non_linear_outputs = Conv2D(filters = out_channels, kernel_size = [size, size],
            padding = 'same', activation = None, 
            kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(second_act_outputs)

    else :
        first_conv_outputs = Conv2D(filters = in_channels // 4, kernel_size = [1, 1], strides = (1, 1), 
            padding = 'same', activation = None, 
            kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(act_outputs)

        second_norm_outputs = BatchNormalization(momentum = norm_momentum)(first_conv_outputs)
        second_act_outputs = ReLU()(second_norm_outputs)
        second_conv_outputs = Conv2D(filters = in_channels // 4, kernel_size = [size, size], 
            strides = (2, 2) if status else (1, 1), padding = 'same', activation = None, 
            kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(second_act_outputs)

        # num_filters = 256 or 512, filter_size = 32 or 16
        third_norm_outputs =BatchNormalization(momentum = norm_momentum)(second_conv_outputs)
        third_act_outputs = ReLU()(third_norm_outputs)
        non_linear_outputs = Conv2D(filters = out_channels, kernel_size = [1, 1], strides =(1, 1),
            padding = 'same', activation = None, 
            kernel_initializer = kernel_init, kernel_regularizer = l2(l2_value))(third_act_outputs)

    if status :
        if zero_pad :
            shortcut_proj = MaxPooling2D(pool_size = (1, 1), strides = (2, 2), padding = 'same')(shortcut_inputs)

            # shortcuts_proj_outputs = [batch_size, image_height, image_width, in_channels]
            # num_channels 부분만 뒤 쪽에 zero padding을 함.
            shortcut_outputs = Lambda(lambda x : pad(x, [[0, 0], [0, 0], [0, 0], [0, out_channels - in_channels]]))(shortcut_proj)
        else :
            shortcut_outputs = Conv2D(filters = out_channels, kernel_size = [1, 1], strides = (2, 2), padding = 'same', 
                activation = None, use_bias = False, 
                kernel_initializer = kernel_init, kernel_regularizer= l2(l2_value))(shortcut_inputs)

        return Add()([non_linear_outputs, shortcut_outputs])
    else :
        return Add()([non_linear_outputs, shortcut_inputs])

def make_pre_activation_model(inputs, *args, **kwargs) :
    num_filters = kwargs["num_filters"]
    kernel_size = kwargs["kernel_size"]
    kernel_init = kwargs["kernel_initializer"]
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
    inputs = first_conv_outputs
    for outer_idx in range(num_box) :
        for inner_idx in range(num_blocks_in_box[outer_idx]) :
            if not outer_idx and not inner_idx :
                outputs = make_layers(inputs, kernel_size, l2_value, norm_momentum, kernel_init,
                    num_filters[outer_idx], num_filters[outer_idx], first_layer = True, bottleneck = bottleneck, zero_pad = zero_pad)
            else :
                outputs = make_layers(inputs, kernel_size, l2_value, norm_momentum, kernel_init,
                    num_filters[outer_idx], num_filters[outer_idx], first_layer = False, bottleneck = bottleneck, zero_pad = zero_pad)

            inputs = outputs
        
        if outer_idx != num_box - 1 : # 마지막 block은 size를 줄이지 않음
            outputs = make_layers(inputs, kernel_size, l2_value, norm_momentum, kernel_init,
                num_filters[outer_idx], num_filters[outer_idx + 1], first_layer = False, bottleneck = bottleneck, zero_pad = zero_pad)
            inputs = outputs            

    final_outputs = ReLU()(inputs)
    avgPool_outputs = GlobalAveragePooling2D()(final_outputs)
    logits = Dense(units = num_categories, activation = 'softmax', 
        kernel_initializer = kernel_init, kernel_regularizer= l2(l2_value))(avgPool_outputs)

    return logits