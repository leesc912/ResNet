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
            outputs = Residual_Block(num_blocks, num_filters, k_init, k_reg, zero_pad, bn_momentum, True)(outputs)
        else :
            outputs = Residual_Block(num_blocks, num_filters, k_init, k_reg, zero_pad, bn_momentum, False)(outputs)
            
        num_filters *= 2

    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    logits = keras.layers.Dense(units = num_category, kernel_initializer = k_init, kernel_regularizer = k_reg)(outputs)

    return keras.Model(inputs = [inputs], outputs = [logits], name = "ResNet")

class Residual_Block(keras.layers.Layer) :
    def __init__(self, num_blocks, num_filters, k_init, k_reg, zero_pad, bn_momentum, down_sampling) :
        super(Residual_Block, self).__init__()

        self.blocks = []
        for idx in range(num_blocks) :
            if down_sampling and not idx : # DownSampling
                self.blocks.append(OneBlock(num_filters, zero_pad, k_init, k_reg, bn_momentum, True))
            else :
                self.blocks.append(OneBlock(num_filters, zero_pad, k_init, k_reg, bn_momentum, False))

    def call(self, inputs) :
        outputs = inputs
        for block in self.blocks :
            outputs = block(outputs)
            
        return outputs

class OneBlock(keras.layers.Layer) :
    # conv -> relu -> bn -> conv -> residual connection -> relu -> bn 
    def __init__(self, num_filters, zero_pad, k_init, k_reg, bn_momentum, down_sampling) :
        super(OneBlock, self).__init__()
        
        if down_sampling :
            self.first_conv = keras.layers.Conv2D(num_filters, 3, 2, padding = "same", use_bias = False,
                                                  kernel_initializer = k_init, kernel_regularizer = k_reg)
            self.down_sampling = DownSampling(num_filters // 2, zero_pad, k_init, k_reg, bn_momentum)
        else :
            self.first_conv = keras.layers.Conv2D(num_filters, 3, 1, padding = "same", use_bias = False,
                                                  kernel_initializer = k_init, kernel_regularizer = k_reg)
            self.down_sampling = None
        
        self.first_bn = keras.layers.BatchNormalization(momentum = bn_momentum)
        self.first_act = keras.layers.ReLU()
        
        self.second_conv = keras.layers.Conv2D(num_filters, 3, 1, padding = "same", use_bias = False,
                                               kernel_initializer = k_init, kernel_regularizer = k_reg)
        self.second_bn = keras.layers.BatchNormalization(momentum = bn_momentum)

        self.second_act = keras.layers.ReLU()

    def call(self, inputs) :
        outputs = self.first_conv(inputs)
        outputs = self.first_bn(outputs)
        outputs = self.first_act(outputs)
        
        outputs = self.second_conv(outputs)
        outputs = self.second_bn(outputs)
        
        if self.down_sampling is not None :
            inputs = self.down_sampling(inputs)

        outputs += inputs # residual connection
        outputs = self.second_act(outputs)

        return outputs
    
class DownSampling(keras.layers.Layer) :
    def __init__(self, num_filters, zero_pad, k_init, k_reg, bn_momentum) :
        super(DownSampling, self).__init__()

        if zero_pad :
            self.first_layer = keras.layers.MaxPool2D((1, 1), strides = (2, 2), padding = "same")
            self.second_layer = keras.layers.Lambda(lambda x : pad(x, [[0, 0], [0, 0], [0, 0], [0, num_filters]]))
        else :
            self.first_layer = keras.layers.Conv2D(num_filters * 2, 1, 2, padding = "same", use_bias = False,
                                                   kernel_initializer = k_init, kernel_regularizer = k_reg)
            self.second_layer = keras.layers.BatchNormalization(momentum = bn_momentum)

    def call(self, inputs) :
        outputs = self.first_layer(inputs)
        outputs = self.second_layer(outputs)
        
        return outputs