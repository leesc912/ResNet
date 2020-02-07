import tensorflow as tf
import tensorflow.keras.backend as K

def test(dataset, model, num_categories) :
    top_k_list = [0, ] * num_categories
    max_k = min(num_categories, 10)
    num_test_data = 0
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