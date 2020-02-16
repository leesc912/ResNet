import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10, cifar100

def augmentation(x, y) :
    x_shape = K.int_shape(x)

    # [40, 40, num_channels]
    x = tf.image.resize_with_crop_or_pad(x, x_shape[0] + 8, x_shape[1] + 8)

    # [32, 32, num_channels]
    x = tf.image.random_crop(x, x_shape)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.per_image_standardization(x)

    return x, y

def preprocessing(x, y) :
    x = tf.image.per_image_standardization(x)
    return x, y

def label_smoothing(labels, num_category, epsilon = 0.1) :
    labels = K.one_hot(labels, num_category)
    return tf.squeeze(((1 - epsilon) * labels) + (epsilon / num_category))

def get_dataset(batch_size, num_category, use_label_smoothing) :
    if num_category == 10 :
        (train_X, train_y), (test_X, test_y) = cifar10.load_data() 
    else :
        (train_X, train_y), (test_X, test_y) = cifar100.load_data()

    train_X = train_X.astype("float32")
    test_X = test_X.astype("float32")
    train_y = train_y.astype("int32")
    test_y = test_y.astype("int32")

    if use_label_smoothing :
        train_y = label_smoothing(train_y, num_category)
        test_y = label_smoothing(test_y, num_category)

    val_boundray = int(len(train_X) * 0.1)
    val_X = train_X[-val_boundray : ]
    val_y = train_y[-val_boundray : ]
    train_X = train_X[ : -val_boundray]
    train_y = train_y[ : -val_boundray]

    num_train = K.int_shape(train_X)[0]
    num_val = K.int_shape(val_X)[0]
    num_test = K.int_shape(test_X)[0]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))

    train_dataset = train_dataset.cache().map(augmentation).shuffle(num_train + 1).batch(batch_size).prefetch(1)
    val_dataset = val_dataset.cache().map(preprocessing).batch(batch_size).prefetch(1)
    test_dataset = test_dataset.cache().map(preprocessing).batch(batch_size).prefetch(1)

    return train_dataset, val_dataset,test_dataset, num_train, num_val, num_test