import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from wide_ResNet import make_wide_ResNet

def plot(save_folder, *args, **kwargs) :
    base_folder = os.path.join(os.getcwd(), save_folder)
    if not os.path.exists(base_folder) :
        os.mkdir(base_folder)

    inputs = tf.keras.Input(shape = (32, 32, 3))

    name = "Wide-ResNet-28-10-B(3,3)"
    logits = make_wide_ResNet(inputs, **kwargs)
    
    model = tf.keras.Model(inputs = inputs, outputs = logits)
    tf.keras.utils.plot_model(model, to_file = os.path.join(base_folder, "{}.png".format(name)))

    # summary 저장
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)

    with open(os.path.join(base_folder, "{}.txt".format(name)), "w", encoding = 'utf-8') as fp :
        fp.write(model_summary)