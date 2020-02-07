import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from models.Original_ResNet import make_original_ResNet
from models.preActivation_ResNet import make_preActivation_ResNet
from models.wide_ResNet import make_wide_ResNet
from Keras_utils.Utils import check_parameters 

def plot(save_folder, *args, **kwargs) :
    check_parameters(**kwargs)

    base_folder = os.path.join(os.getcwd(), save_folder)
    if not os.path.exists(base_folder) :
        os.mkdir(base_folder)

    inputs = tf.keras.Input(shape = (32, 32, 3))
    
    model_structure = kwargs["model_structure"]
    num_layers = kwargs["num_layers"]

    if model_structure == 'original' :
        name = "Original-ResNet-{}_ZeroPad-{}".format(
            num_layers, kwargs['zero_pad']
        )
        logits = make_original_ResNet(inputs, **kwargs)
    elif model_structure == 'pre' :
        name = "Pre-Activation-ResNet-{}_Bottleneck-{}".format(
            num_layers, kwargs['bottleneck']
        )
        logits = make_preActivation_ResNet(inputs, **kwargs)
    else :
        name = "Wide-ResNet-{}-KerenlWidth-{}-B(3, 3)".format(
            num_layers, kwargs['kernel_width']
        )
        logits = make_wide_ResNet(inputs, **kwargs)

    model = tf.keras.Model(inputs = inputs, outputs = logits)
    tf.keras.utils.plot_model(model, to_file = os.path.join(base_folder, "{}.png".format(name)))

    # summary 저장
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)

    with open(os.path.join(base_folder, "{}.txt".format(name)), "w", encoding = 'utf-8') as fp :
        fp.write(model_summary)