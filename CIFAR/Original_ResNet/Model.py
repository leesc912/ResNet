from tensorflow import pad, keras

def make_resnet(num_category, num_layers, bn_momentum, shortcut) :
    num_blocks = (num_layers - 2) // (3 * 2)

    k_init = keras.initializers.TruncatedNormal(stddev = 0.02)
    k_reg = keras.regularizers.l2(0.0001)

    inputs = keras.Input(shape = (32, 32, 3, ), name = "Inputs")

    outputs = block(inputs, 16, 3, 1, k_init, k_reg, bn_momentum = bn_momentum)

    in_filters = out_filters = 16
    for idx in range(3) :
            outputs = make_residual_block(outputs, num_blocks, in_filters, out_filters, k_init, k_reg,
                                          bn_momentum, shortcut, idx == 0)
            in_filters = out_filters
            out_filters *= 2

    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    logits = keras.layers.Dense(units = num_category, kernel_initializer = k_init, kernel_regularizer = k_reg)(outputs)

    return keras.Model(inputs = [inputs], outputs = [logits], name = "ResNet-{}".format(num_layers))

def make_residual_block(inputs, num_blocks, in_filters, out_filters, k_init, k_reg, bn_momentum, shortcut, first_layer) :
    outputs = inputs
    for idx in range(num_blocks) :
        if not first_layer and not idx :
            outputs = residual_connection(outputs, in_filters, out_filters, k_init, k_reg, bn_momentum, shortcut)
        else :
            outputs = residual_connection(outputs, out_filters, out_filters, k_init, k_reg, bn_momentum, shortcut)
            
    return outputs

def residual_connection(inputs, in_filters, out_filters, k_init, k_reg, bn_momentum, shortcut) :
    if in_filters != out_filters :
        outputs = block(inputs, out_filters, 3, 2, k_init, k_reg, bn_momentum = bn_momentum)
        if shortcut == "identity" :
            inputs = identity_shortcut(inputs, out_filters - in_filters)
        else :
            inputs = projection_shortcut(inputs, out_filters, k_init, k_reg, bn_momentum = bn_momentum)
    else :
         outputs = block(inputs, out_filters, 3, 1, k_init, k_reg, bn_momentum = bn_momentum)
    outputs = block(outputs, out_filters, 3, 1, k_init, k_reg, use_act = False, bn_momentum = bn_momentum)
        
    outputs += inputs
    return keras.layers.ReLU()(outputs)
    
def identity_shortcut(inputs, num_filters) :
    outputs = keras.layers.MaxPool2D((1, 1), strides = (2, 2), padding = "same")(inputs)
    outputs = keras.layers.Lambda(lambda x : pad(x, [[0, 0], [0, 0], [0, 0], [0, num_filters]]))(outputs)
    
    return outputs

def projection_shortcut(inputs, num_filters, k_init, k_reg, use_bn = True, bn_momentum = None) :
    outputs = keras.layers.Conv2D(num_filters, 1, 2, padding = "same", use_bias = False,
                                  kernel_initializer = k_init, kernel_regularizer = k_reg)(inputs)
    if use_bn :
        assert bn_momentum is not None
        outputs = keras.layers.BatchNormalization(momentum = bn_momentum)(outputs)
    
    return outputs

def block(inputs, num_filters, size, strides, k_init, k_reg, 
          use_act = True, use_bn = True, bn_momentum = None) :
    outputs = keras.layers.Conv2D(num_filters, size, strides, padding = "same", use_bias = False,
                                 kernel_initializer = k_init, kernel_regularizer = k_reg)(inputs)
    if use_bn :
        assert bn_momentum is not None
        outputs = keras.layers.BatchNormalization(momentum = bn_momentum)(outputs)
    if use_act :
        outputs = keras.layers.ReLU()(outputs)
        
    return outputs