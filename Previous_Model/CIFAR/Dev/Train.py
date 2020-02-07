import os
from datetime import datetime

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import image

from models.Original_ResNet import make_original_ResNet
from models.preActivation_ResNet import make_preActivation_ResNet
from models.wide_ResNet import make_wide_ResNet

from Callbacks.LearningRateScheduler import SpecificLearningRateScheduler
from Callbacks.Callback import RecordCallbacks, custom_in_top_k_metrics
from Keras_utils.Utils import save_model_info, load_checkpoint, check_parameters
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
    if isinstance(kwargs["epochs"], int) and kwargs["epochs"] > 0 :
        check_parameters(**kwargs)

        batch_size          = 128
        val_split_ratio     = 0.1
        num_categories      = kwargs["num_categories"]

        if kwargs["folder"] is None :
            folder = os.path.join(".", "cifar_result")
        else :
            folder = kwargs["folder"]
        if not os.path.exists(folder) :
            os.mkdir(folder)

        save_folder = os.path.join(folder, str(datetime.now().strftime('%Y-%m-%d__%H%M%S')))
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
        
        val_boundray = int(len(train_X) * val_split_ratio)
        val_X = train_X[-val_boundray : ]
        val_y = train_y[-val_boundray : ]
        train_X = train_X[ : -val_boundray]
        train_y = train_y[ : -val_boundray]

        trainData_shape = K.int_shape(train_X)
        valData_shape = K.int_shape(val_X)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).map(augmentation).shuffle(trainData_shape[0]).batch(128)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y)).map(augmentation).batch(128)

        # Model
        if kwargs["ckpt_model_path"] is not None :
            print("\n\nLoading model from checkpoint...\n\n")
            model, ckpt_lr, ckpt_epoch = load_checkpoint(**kwargs)
        else :
            print("\n\nMaking new model...\n\n")

            inputs = tf.keras.Input(shape = tuple(trainData_shape[1 : ]))
            if kwargs["model_structure"] == 'original' :
                logits = make_original_ResNet(inputs, **kwargs)
            elif kwargs["model_structure"] == 'pre' :
                logits = make_preActivation_ResNet(inputs, **kwargs)
            else :
                logits = make_wide_ResNet(inputs, **kwargs)
            model = tf.keras.Model(inputs = inputs, outputs = logits)


        # Learning Rate
        lr = kwargs["initial_lr"]
        if lr is not None :
            # checkpoint의 learning rate를 사용하지 않거나, 새로운 model을 만들었을 때
            lr = tf.Variable(lr, trainable = False)
        elif (kwargs["ckpt_model_path"] is not None) and (ckpt_lr is not None) :
            # learning rate가 None이고, checkpoint의 learning rate를 사용하고 싶을 때
            lr = tf.Variable(ckpt_lr, trainable = False)
        else :
            # learning rate가 None이고, checkpoint에 learning rate가 존재하지 않을 때
            raise Exception("\nLearning rate is None\n")


        # kwargs['initial_epoch']
        initial_epoch = kwargs["initial_epoch"]
        if initial_epoch is not None :
            # checkpoint의 epoch를 사용하지 않거나, 새로운 model을 만들었을 때
            pass
        elif (kwargs["ckpt_model_path"] is not None) and (ckpt_epoch is not None) :
            # checkpoint의 epoch를 사용하고 싶을 때
            initial_epoch =  ckpt_epoch
        else :
            initial_epoch = 0


        # Compile
        if kwargs["model_structure"] in ['original', 'pre'] :
            model.compile(
                optimizer = tf.keras.optimizers.SGD(
                    lr = lr, momentum = 0.9),
                loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']
            )
        else :
            model.compile(
                optimizer = tf.keras.optimizers.SGD(
                    lr = lr, momentum = 0.9, nesterov = True),
                loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']
            )

        save_model_info(model, save_folder, model_structure_path, model_weights_path, **kwargs)
        

        # Model callback
        callback_list = [
            RecordCallbacks(model_weights_path, model_result_file, ckpt_info_file),
            custom_in_top_k_metrics(val_dataset, top_k_record_file, 
                batch_size = batch_size, num_categories = num_categories, max_in_top_k = 10, num_data = valData_shape[0])
        ]

        if kwargs["lr_scheduler"] == True :
            epoch = initial_epoch
            callback_list += [SpecificLearningRateScheduler(epoch, **kwargs)]


        # Training
        history = model.fit(train_dataset,
            epochs = kwargs["epochs"],
            shuffle = False,
            validation_data = val_dataset,
            callbacks = callback_list,
            verbose = 1,
            initial_epoch = initial_epoch)

    else :
        test(**kwargs)