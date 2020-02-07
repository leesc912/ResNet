import os
import tensorflow as tf
import tensorflow.keras.backend as K
from datetime import datetime

from models.Original_ResNet import make_original_ResNet
from models.preActivation_ResNet import make_preActivation_ResNet
from models.wide_ResNet import make_wide_ResNet

from Callbacks.LearningRateScheduler import SpecificLearningRateScheduler
from Callbacks.Callback import RecordCallbacks, custom_in_top_k_metrics

from Keras_utils.Utils import save_model_info, load_checkpoint, check_parameters
from Keras_utils.data_generator import CustomImageDataGenerator

from Test import test

tf.enable_eager_execution()

def train(**kwargs) :
    if isinstance(kwargs["epochs"], int) and kwargs["epochs"] > 0 :
        check_parameters(**kwargs)

        batch_size          = 128
        val_split_ratio     = 0.1
        momentum            = 0.9
        num_categories      = kwargs["num_categories"]

        if kwargs["base_folder"] is None :
            base_folder = os.path.join(".", "cifar_result")
        else :
            base_folder = kwargs["base_folder"]
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
        
        val_boundray = int(len(train_X) * val_split_ratio)
        val_X = train_X[-val_boundray : ]
        val_y = train_y[-val_boundray : ]
        train_X = train_X[ : -val_boundray]
        train_y = train_y[ : -val_boundray]

        trainData_shape = K.int_shape(train_X)
        num_trainData = trainData_shape[0]

        # CustomImageDataGenerator
        train_generator = CustomImageDataGenerator(train_X, train_y, trainData_shape[1 : ], True, 
            batch_size, num_categories, True) 
        val_generator = CustomImageDataGenerator(val_X, val_y, trainData_shape[1 : ], False, 
            batch_size, num_categories, False)


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
        if kwargs["model_structure"] in ['basic', 'pre'] :
            model.compile(
                optimizer = tf.keras.optimizers.SGD(
                    lr = lr, momentum = momentum),
                loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']
            )
        else :
            model.compile(
                optimizer = tf.keras.optimizers.SGD(
                    lr = lr, momentum = momentum, nesterov = True),
                loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']
            )

        save_model_info(model, save_folder, model_structure_path, model_weights_path, **kwargs)
        

        # Model callback
        callback_list = [RecordCallbacks(model_weights_path, model_result_file, ckpt_info_file)]

        callback_list += [
            custom_in_top_k_metrics(val_generator, top_k_record_file, 
                batch_size = batch_size, num_categories = num_categories, max_in_top_k = 10)
        ]

        if kwargs["lr_scheduler"] == True :
            epoch = initial_epoch
            callback_list += [SpecificLearningRateScheduler(epoch, **kwargs)]


        # Training
        history = model.fit_generator(train_generator,
            epochs = kwargs["epochs"],
            shuffle = False,
            validation_data = val_generator,
            callbacks = callback_list,
            verbose = 1,
            initial_epoch = initial_epoch)


        # Evaluation
        print("\n\nModel Evaluation")
        test_generator = CustomImageDataGenerator(test_X, test_y, trainData_shape[1 : ], False, 
            batch_size, num_categories, False)
        model.evaluate_generator(test_generator, verbose = 1)

    else :
        test(**kwargs)