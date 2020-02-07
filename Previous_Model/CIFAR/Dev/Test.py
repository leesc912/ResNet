import tensorflow as tf
from tensorflow import image
import tensorflow.keras.backend as K

from Keras_utils.Utils import load_checkpoint

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

def test(**kwargs) :
    num_categories = kwargs['num_categories']
    if num_categories == 10 :
        (_, _), (test_X, test_y) = tf.keras.datasets.cifar10.load_data() 
    else :
        (_, _), (test_X, test_y) = tf.keras.datasets.cifar100.load_data()

    model, _, _ = load_checkpoint(**kwargs)

    # Evaluation
    dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).map(augmentation).batch(128)

    top_k_list = [0, ] * num_categories
    max_k = min(num_categories, 10)
    num_test_data = 0

    print("\nTesting...\n")
    for outer_idx, (inputs, labels) in enumerate(dataset) :
        num_test_data += K.int_shape(inputs)[0]

        # y_pred shape : (batch_size, num_categories) (after softmax activation)
        y_pred = model.predict_on_batch(inputs)

        # softmax value가 가장 큰 index부터 앞에 위치
        sorted_y_pred = tf.argsort(y_pred, -1, 'DESCENDING')
        y_true_location = tf.where(tf.math.equal(labels, sorted_y_pred))

        indicies = K.get_value(y_true_location[ : , 1])

        for inner_idx in indicies :
            top_k_list[inner_idx] += 1

    accumulated_list = [0, ] * max_k
    accumulated_list[0] = top_k_list[0]

    for idx in range(1, max_k) :
        accumulated_list[idx] = accumulated_list[idx - 1] + top_k_list[idx]

    str_ = "\nin-top-k Accuracy\n"
    str_ += "  |  ".join([" Top-{}".format(idx + 1) if idx != 9 else "Top-{}".format(idx + 1)
            for idx in range(max_k)]) + "\n"
    str_ += "  |  ".join(
        ["{:.4f}".format(accumulated_list[idx] / num_test_data) for idx in range(max_k)]
    ) + "\n\n"

    print(str_)