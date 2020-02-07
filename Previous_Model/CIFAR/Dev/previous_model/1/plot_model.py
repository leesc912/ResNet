import tensorflow as tf
from make_model import make_resnet_model
import os

def plot(save_folder, *args, **kwargs) :
    base_folder = os.path.join(os.getcwd(), save_folder)
    if not os.path.exists(base_folder) :
        os.mkdir(base_folder)

    inputs = tf.keras.Input(shape = (32, 32, 3))
    
    num_blocks = kwargs["num_blocks_in_box"]
    nums = 3 if kwargs["bottleneck"] else 2
    total_layers = 0
    for num in num_blocks :
        total_layers += num * nums
    total_layers += 2

    name = "ResNet-{}_PreActivation-{}_BottleNeck-{}_ZeroPad-{}".format(
        total_layers, kwargs['pre_activation'], kwargs['bottleneck'], kwargs['zero_pad']
    )
    logits = make_resnet_model(inputs, **kwargs)
    model = tf.keras.Model(inputs = inputs, outputs = logits)
    tf.keras.utils.plot_model(model, to_file = os.path.join(base_folder, "{}.png".format(name)))

    # summary 저장
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)

    with open(os.path.join(base_folder, "{}.txt".format(name)), "w", encoding = 'utf-8') as fp :
        fp.write(model_summary)