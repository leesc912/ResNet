import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Flatten

from Keras_utils.data_generator import CustomImageDataGenerator
from Keras_utils.Utils import save_model_info, load_checkpoint

tf.enable_eager_execution()

def test(**kwargs) :
    num_categories = kwargs['num_categories']
    if num_categories == 10 :
        (_, _), (test_X, test_y) = tf.keras.datasets.cifar10.load_data() 
    else :
        (_, _), (test_X, test_y) = tf.keras.datasets.cifar100.load_data()

    model, _, _ = load_checkpoint(**kwargs)

    # Evaluation
    test_generator = CustomImageDataGenerator(test_X, test_y, K.int_shape(test_X)[1 : ], False, 
        128, kwargs["num_categories"], False)
    
    num_batches = len(test_generator)
    num_test_data = test_generator.get_num_data()
   
    top_k_list = [0, ] * num_categories
    max_k = min(num_categories, 10)

    print("\nTesting...\n")
    for idx in range(num_batches) :
        # self.val_gen[idx] => self.val_gen.__getitem__[idx]
        val_X, y_true = test_generator[idx]

        # y_pred shape : (batch_size, num_categories) (after softmax activation)
        y_pred = model.predict_on_batch(val_X)

        # softmax value가 가장 큰 index부터 앞에 위치
        sorted_y_pred = tf.argsort(y_pred, -1, 'DESCENDING')
        y_true_location = tf.where(tf.math.equal(y_true, sorted_y_pred))

        indicies = K.get_value(y_true_location[ : , 1])

        for idx in indicies :
            top_k_list[idx] += 1

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