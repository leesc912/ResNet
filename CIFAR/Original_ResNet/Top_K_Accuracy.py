import csv
import tensorflow as tf
from tensorflow.keras import backend as K

class custom_top_k_metrics() :
    # Top-1 ~ Top-k 까지의 정확도 누적값을 출력
    def __init__(self, model, dataset, record_file, num_category, max_in_top_k, num_data, use_label_smoothing) :
        self.model = model
        self.dataset = dataset

        self.record_file = record_file
        self.num_category = num_category
        self.max_k = min(10, max_in_top_k)
        self.num_data = num_data

        self.use_label_smoothing = use_label_smoothing
        if self.use_label_smoothing :
            self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
            self.val_function = tf.function(self.forward, input_signature = [tf.TensorSpec((None, 32, 32, 3), tf.float32), 
                                                                             tf.TensorSpec((None, self.num_category), tf.float32)])
        else :
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
            self.val_function = tf.function(self.forward, input_signature = [tf.TensorSpec((None, 32, 32, 3), tf.float32), 
                                                                             tf.TensorSpec((None, 1), tf.int32)])

        self.val_loss_metric = tf.keras.metrics.Mean(name = "val_Loss")

        self.csv_header = ["Epoch"] + ["Top {}".format(idx + 1) for idx in range(self.max_k)]
        with self.record_file.open("w", newline = '', encoding = "utf-8") as fp :
            writer = csv.DictWriter(fp, fieldnames = self.csv_header)
            writer.writeheader()

    def evaluate(self, epoch) :
        self.val_loss_metric.reset_states()

        top_k_list = [0, ] * self.num_category
        for X, y_true in self.dataset :
            num_inputs = K.int_shape(X)[0]

            # y_pred shape : (batch_size, num_categories)
            y_pred = self.val_function(X, y_true)

            if self.use_label_smoothing :
                y_true = tf.expand_dims(tf.math.argmax(y_true, axis = -1, output_type = tf.int32), axis = 1)

            # value가 가장 큰 index부터 앞에 위치
            sorted_y_pred = tf.argsort(y_pred, -1, 'DESCENDING')
            y_true_location = tf.where(tf.math.equal(y_true, sorted_y_pred))

            indicies = K.get_value(y_true_location[ : , 1])

            for inner_idx in indicies :
                top_k_list[inner_idx] += 1

        accumulated_list = [0, ] * self.max_k
        accumulated_list[0] = top_k_list[0]

        for idx in range(1, self.max_k) :
            accumulated_list[idx] = accumulated_list[idx - 1] + top_k_list[idx]
        accumulated_list = [num / self.num_data for num in accumulated_list]

        with self.record_file.open("a+", newline = '', encoding = "utf-8") as fp :
            writer = csv.DictWriter(fp, fieldnames = self.csv_header)
            value_dic = {"Top {}".format(idx + 1) : "{:.4f}".format(accumulated_list[idx]) for idx in range(self.max_k)}
            value_dic["Epoch"] = epoch
            writer.writerow(value_dic)

        # loss와 Top 1 accuracy 반환
        return self.val_loss_metric.result(), accumulated_list[0]

    def forward(self, inputs, labels) :
        logits = self.model(inputs, training = False)
        loss = self.loss_function(labels, logits)

        self.val_loss_metric.update_state(loss)

        return logits