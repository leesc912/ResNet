from tensorflow import pad, keras

def make_resnet(num_category, num_layers, zero_pad, bn_momentum) :
    num_blocks = (num_layers - 2) // 6
    num_filters = 16

    k_init = keras.initializers.TruncatedNormal(stddev = 0.02)
    k_reg = keras.regularizers.l2(0.0001)

    inputs = keras.Input(shape = (32, 32, 3, ), name = "Inputs")

    # ResNet의 첫 번째 부분
    outputs = keras.layers.Conv2D(num_filters, 3, 2, padding = "same", use_bias = False,
                                  kernel_initializer = k_init, kernel_regularizer = k_reg)(inputs)
    outputs = keras.layers.BatchNormalization(momentum = bn_momentum)(outputs)
    outputs = keras.layers.ReLU()(outputs)

    for idx in range(3) :
        if idx :
            outputs = make_residual_block(outputs, num_blocks, num_filters, k_init, k_reg, zero_pad, bn_momentum, True)
        else :
            outputs = make_residual_block(outputs, num_blocks, num_filters, k_init, k_reg, zero_pad, bn_momentum, False)
            
        num_filters *= 2

    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    logits = keras.layers.Dense(units = num_category, kernel_initializer = k_init, kernel_regularizer = k_reg)(outputs)

    return keras.Model(inputs = [inputs], outputs = [logits], name = "ResNet")

def make_residual_block(inputs, num_blocks, num_filters, k_init, k_reg, zero_pad, bn_momentum, down_sampling) :
    outputs = inputs
    for idx in range(num_blocks) :
        if down_sampling and not idx :
            outputs = make_one_block(outputs, num_filters, zero_pad, k_init, k_reg, bn_momentum, True)
        else :
            outputs = make_one_block(outputs, num_filters, zero_pad, k_init, k_reg, bn_momentum, False)

    return outputs

def make_one_block(inputs, num_filters, zero_pad, k_init, k_reg, bn_momentum, down_sampling) :
    if down_sampling :
        outputs = keras.layers.Conv2D(num_filters, 3, 2, padding = "same", use_bias = False,
                                    kernel_initializer = k_init, kernel_regularizer = k_reg)(inputs)
        inputs = shortcut_sampling(inputs, num_filters // 2, zero_pad, k_init, k_reg, bn_momentum)
    else :
        outputs = keras.layers.Conv2D(num_filters, 3, 1, padding = "same", use_bias = False,
                                    kernel_initializer = k_init, kernel_regularizer = k_reg)(inputs)        
    outputs = keras.layers.BatchNormalization(momentum = bn_momentum)(outputs)
    outputs = keras.layers.ReLU()(outputs)

    outputs = keras.layers.Conv2D(num_filters, 3, 1, padding = "same", use_bias = False,
                                  kernel_initializer = k_init, kernel_regularizer = k_reg)(outputs)
    outputs = keras.layers.BatchNormalization(momentum = bn_momentum)(outputs)

    outputs += inputs
    outputs = keras.layers.ReLU()(outputs)

    return outputs

def shortcut_sampling(inputs, num_filters, zero_pad, k_init, k_reg, bn_momentum) :
    if zero_pad :
        outputs = keras.layers.MaxPool2D((1, 1), strides = (2, 2), padding = "same")(inputs)
        outputs = keras.layers.Lambda(lambda x : pad(x, [[0, 0], [0, 0], [0, 0], [0, num_filters]]))(outputs)
    else :
        outputs = keras.layers.Conv2D(num_filters * 2, 1, 2, padding = "same", use_bias = False,
                                      kernel_initializer = k_init, kernel_regularizer = k_reg)(inputs)
        outputs = keras.layers.BatchNormalization(momentum = bn_momentum)(outputs)

    return outputs