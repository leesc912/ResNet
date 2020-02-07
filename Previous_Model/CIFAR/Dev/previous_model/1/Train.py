import os
import tensorflow as tf
import tensorflow.keras.backend as K
from datetime import datetime

from make_model import make_resnet_model
from Callback import RecordCallbacks, custom_in_top_k_metrics
from LearningRateScheduler import SpecificLearningRateScheduler, ConditionalLearningRateScheduler
from data_generator import CustomImageDataGenerator
from Utils import save_model_info, load_checkpoint
from Test import test

tf.enable_eager_execution()

def train(*args, **kwargs) :
    if isinstance(kwargs["epochs"], int) and kwargs["epochs"] > 0 :
        base_folder = os.path.join(".", "cifar_result")
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
    
        if kwargs['num_categories'] == 10 :
            (train_X, train_y), (test_X, test_y) = tf.keras.datasets.cifar10.load_data() 
        else :
            (train_X, train_y), (test_X, test_y) = tf.keras.datasets.cifar100.load_data() 
        
        val_boundray = int(len(train_X) * kwargs['val_split_ratio'])
        val_X = train_X[-val_boundray : ]
        val_y = train_y[-val_boundray : ]
        train_X = train_X[ : -val_boundray]
        train_y = train_y[ : -val_boundray]

        trainData_shape = K.int_shape(train_X)
        num_trainData = trainData_shape[0]

        # CustomImageDataGenerator
        train_generator = CustomImageDataGenerator(train_X, train_y, trainData_shape[1 : ], True, 
            kwargs["batch_size"], kwargs["num_categories"], kwargs["data_augmentation"]) 
        val_generator = CustomImageDataGenerator(val_X, val_y, trainData_shape[1 : ], False, 
            kwargs["batch_size"], kwargs["num_categories"], False)


        # Model
        if kwargs["ckpt_model_path"] is not None :
            print("\n\nLoading model from checkpoint...\n\n")
            model, ckpt_lr, ckpt_epoch = load_checkpoint(**kwargs)
        else :
            print("\n\nMaking new model...\n\n")
            inputs = tf.keras.Input(shape = tuple(trainData_shape[1 : ]))
            logits = make_resnet_model(inputs, **kwargs)
            model = tf.keras.Model(inputs = inputs, outputs = logits)


        # Learning Rate
        lr = kwargs["base_lr"]
        if lr is not None :
            # checkpoint의 learning rate를 사용하지 않거나, 새로운 model을 만들었을 때
            lr = tf.Variable(lr, trainable = False)
        elif (kwargs["ckpt_model_path"] is not None) and (ckpt_lr is not None) :
            # learning rate가 None이고, checkpoint의 learning rate를 사용하고 싶을 때
            lr = tf.Variable(ckpt_lr, trainable = False)
        else :
            # learning rate가 None이고, checkpoint에 learning rate가 존재하지 않을 때
            raise Exception("\nLearning rate is None\n")


        # kwargs['start_epochs']
        initial_epoch = kwargs["start_epochs"]
        if initial_epoch not in [None, 0] :
            # checkpoint의 epoch를 사용하지 않거나, 새로운 model을 만들었을 때
            pass
        elif (kwargs["ckpt_model_path"] is not None) and (ckpt_epoch is not None) :
            # checkpoint의 epoch를 사용하고 싶을 때
            initial_epoch =  ckpt_epoch
        else :
            initial_epoch = 0


        # Compile
        if kwargs["opt_type"] == 'SGD' :
            model.compile(optimizer = tf.keras.optimizers.SGD(lr = lr, momentum = kwargs["SGD_momentum"]),
                loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        elif kwargs["opt_type"] == 'Adam' :
            model.compile(optimizer = tf.keras.optimizers.Adam(lr = lr),
                loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

        save_model_info(model, save_folder, model_structure_path, model_weights_path, kwargs)
        

        # Model callback
        callback_list = [RecordCallbacks(model_weights_path, model_result_file, ckpt_info_file)]
        
        if kwargs["print_in_top_k_acc"] :
            callback_list += [
                custom_in_top_k_metrics(val_generator, top_k_record_file, **kwargs)]

        if kwargs["lr_scheduler"] == "CLRS" :
            callback_list += [
                ConditionalLearningRateScheduler(**kwargs)]
        elif kwargs["lr_scheduler"] == "SLRC" :
            callback_list += [
                SpecificLearningRateScheduler(**kwargs)]


        # Training
        history = model.fit_generator(train_generator,
            steps_per_epoch = num_trainData / kwargs["batch_size"], epochs = kwargs["epochs"],
            shuffle = False,
            validation_data = val_generator,
            callbacks = callback_list,
            verbose = kwargs["verbose"],
            initial_epoch = initial_epoch)


        # Evaluation
        print("\n\nModel Evaluation")
        test_generator = CustomImageDataGenerator(test_X, test_y, trainData_shape[1 : ], False, 
            kwargs["batch_size"], kwargs["num_categories"], False)
        model.evaluate_generator(test_generator, verbose = kwargs["verbose"])

    else :
        test(**kwargs)