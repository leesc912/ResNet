import os
from datetime import datetime

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import image

from preActivation_ResNet import make_preActivation_ResNet
from Callback import RecordCallbacks, custom_in_top_k_metrics
from Utils import save_model_info
from Test import test

def augmentation(x, y) :
    x_shape = K.int_shape(x)

    # [40, 40, num_channels]
    x = tf.dtypes.cast(x, tf.float32)
    x = image.resize_with_crop_or_pad(x, x_shape[0] + 8, x_shape[1] + 8)

    # [32, 32, num_channels]
    x = image.random_crop(x, x_shape)
    x = image.random_flip_left_right(x)
    x = image.per_image_standardization(x)

    y = tf.dtypes.cast(y, tf.int32)
    
    return x, y

def train(**kwargs) :
    num_categories = kwargs["num_categories"]

    if kwargs["folder"] is None :
        base_folder = os.path.join(".", "cifar_result")
    else :
        base_folder = kwargs["folder"]
    if not os.path.exists(base_folder) :
        os.mkdir(base_folder)

    save_folder = os.path.join(base_folder, str(datetime.now().strftime('%Y-%m-%d__%H%M%S')))
    os.mkdir(save_folder)
        
    save_model_path = os.path.join(save_folder, "model")
    os.mkdir(save_model_path)    
        
    model_weights_path = os.path.join(save_model_path, "weights")
    os.mkdir(model_weights_path)
        
    model_structure_path = os.path.join(save_model_path, "structure")
    os.mkdir(model_structure_path)
        
    model_result_file = os.path.join(save_folder, "model_result.txt")
    ckpt_info_file = os.path.join(model_weights_path, "model-ckpt-info.json")
    top_k_record_file = os.path.join(save_folder, "in-top-k-record.txt")
    
    if num_categories == 10 :
        (train_X, train_y), (test_X, test_y) = tf.keras.datasets.cifar10.load_data() 
    else :
        (train_X, train_y), (test_X, test_y) = tf.keras.datasets.cifar100.load_data() 
        
    val_boundray = int(len(train_X) * 0.1)
    val_X = train_X[-val_boundray : ]
    val_y = train_y[-val_boundray : ]
    train_X = train_X[ : -val_boundray]
    train_y = train_y[ : -val_boundray]

    trainData_shape = K.int_shape(train_X)
    valData_shape = K.int_shape(val_X)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).map(augmentation).shuffle(trainData_shape[0]).batch(128)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y)).map(augmentation).batch(128)

    inputs = tf.keras.Input(shape = tuple(trainData_shape[1 : ]))
    logits = make_preActivation_ResNet(inputs, **kwargs)
    model = tf.keras.Model(inputs = inputs, outputs = logits)

    model.compile(optimizer = tf.keras.optimizers.SGD(
        lr = tf.Variable(0.01, trainable = False), momentum = 0.9),
        loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    save_model_info(model, save_folder, model_structure_path, model_weights_path, **kwargs)
        
    # Model callback
    callback_list = [
        RecordCallbacks(model_weights_path, model_result_file, ckpt_info_file),
        custom_in_top_k_metrics(val_dataset, top_k_record_file, 
            batch_size = 128, num_categories = num_categories, max_in_top_k = 10, num_data = valData_shape[0])
    ]

    # Training
    history = model.fit(train_dataset, 
        epochs = 5,
        shuffle = False,
        validation_data = val_dataset,
        callbacks = callback_list,
        verbose = 1,
        initial_epoch = 0
    )

    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).map(augmentation).batch(128)
    model.evaluate(test_dataset, verbose = 1)
    test(test_dataset, model, num_categories)