import os
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from datetime import datetime

class custom_in_top_k_metrics(tf.keras.callbacks.Callback) :
    '''
        Top-1 ~ Top-k 까지의 정확도 누적값을 출력
    '''
    def __init__(self, dataset, record_log, **kwargs) :
        super(custom_in_top_k_metrics, self).__init__()

        self.dataset = dataset
        self.record_log = record_log

        self.batch_size = kwargs["batch_size"]
        self.num_categories = kwargs["num_categories"]
        self.max_k = min(10, kwargs["max_in_top_k"])
        self.num_data = kwargs["num_data"]

        str_ = "\nin-top-k Accuracy\n"
        str_ += "  |  ".join([" Top-{}".format(idx + 1) if idx != 9 else "Top-{}".format(idx + 1)
                for idx in range(self.max_k)]) + "\n"

        with open(self.record_log, "a+", encoding = 'utf-8') as fp :
            fp.write(str_)

    def on_epoch_end(self, epoch, logs = {}) :
        top_k_list = [0, ] * self.num_categories
        
        for outer_idx, (inputs, labels) in enumerate(self.dataset) :
            num_inputs = K.int_shape(inputs)[0]

            # y_pred shape : (batch_size, num_categories) (after softmax activation)
            y_pred = self.model.predict_on_batch(inputs)

            # softmax value가 가장 큰 index부터 앞에 위치
            sorted_y_pred = tf.argsort(y_pred, -1, 'DESCENDING')
            y_true_location = tf.where(tf.math.equal(labels, sorted_y_pred))

            indicies = K.get_value(y_true_location[ : , 1])

            for inner_idx in indicies :
                top_k_list[inner_idx] += 1

        accumulated_list = [0, ] * self.max_k
        accumulated_list[0] = top_k_list[0]

        for idx in range(1, self.max_k) :
            accumulated_list[idx] = accumulated_list[idx - 1] + top_k_list[idx]

        str_ = "  |  ".join(
            ["{:.4f}".format(accumulated_list[idx] / self.num_data) for idx in range(self.max_k)]
        )

        with open(self.record_log, "a+", encoding = 'utf-8') as fp :
            fp.write(str_ + "\n")
            
class RecordCallbacks(tf.keras.callbacks.Callback) :
    def __init__(self, model_weights_path, record_log, ckpt_info_file) :
        super(RecordCallbacks, self).__init__()

        self.model_weight_path = model_weights_path
        self.record_log = record_log
        self.ckpt_info_file = ckpt_info_file

        self.make_dict()
        
        self.epoch_start_time = 0

    def on_epoch_begin(self, epoch, logs = None) :
        self.epoch_start_time = datetime.now()

    def on_epoch_end(self, epoch, logs = {}) :
        epoch_end_time = datetime.now()

        t_acc = float(logs['accuracy'])
        t_loss = float(logs['loss'])
        val_acc = float(logs['val_accuracy'])
        val_loss = float(logs['val_loss'])

        cur_lr = float(K.get_value(self.model.optimizer.lr))
        fname = "epoch-{}_ValAcc-{:0.5f}_ValLoss-{:0.5f}_LR-{}.h5".format(epoch, val_acc, val_loss, cur_lr)

        self.update_dict(epoch, fname, cur_lr, val_acc, val_loss)
        self.save_json_file(self.ckpt_info_file, 
            {'max_acc_ckpt' : self.max_acc_dict, 'min_loss_ckpt' : self.min_loss_dict, 'the_latest_ckpt' : self.the_latest_dict})

        self.model.save_weights(os.path.join(self.model_weight_path, fname))

        with open(self.record_log, "a+", encoding = 'utf-8') as fp :
            str_ = "Epoch = [{:5d}] - Learning Rate = [{}] - End Time [ {} ]\n".format(
                epoch, cur_lr, str(epoch_end_time.strftime('%Y-%m-%d %H:%M:%S')))
            str_ += "Elapsed Time = {}\n".format(epoch_end_time - self.epoch_start_time)
            str_ += "Training         =>   Accuracy = [{:8.5f}]  Loss     = [{:8.5f}]\n".format(t_acc, t_loss)
            str_ += "Testing          =>   Accuracy = [{:8.5f}]  Loss     = [{:8.5f}]\n".format(val_acc, val_loss)
            str_ += "Maximum Accuracy =>   Epoch    = [{:8d}]  Accuracy = [{:8.5f}]\n".format(self.max_acc_epoch, self.max_acc_dict['value'])
            str_ += "Minimum Loss     =>   Epoch    = [{:8d}]  Loss     = [{:8.5f}]\n\n".format(self.min_loss_epoch, self.min_loss_dict['value'])
            str_ += " - " * 15 + "\n\n"
            fp.write(str_)

    def update_dict(self, epoch, fname, cur_lr, val_acc, val_loss) :
        if self.max_acc_dict['value'] < val_acc :
            self.max_acc_epoch = epoch
            self.max_acc_dict['value'] = val_acc
            self.max_acc_dict['path'] = fname

        if self.min_loss_dict['value'] > val_loss :
            self.min_loss_epoch = epoch
            self.min_loss_dict['value'] = val_loss
            self.min_loss_dict['path'] = fname

        self.the_latest_dict['path'] = fname

    def make_dict(self) :
        self.max_acc_dict = dict()
        self.min_loss_dict = dict()
        self.the_latest_dict = dict()

        self.max_acc_dict['value'] = 0
        self.min_loss_dict['value'] = 1e10
        
    def save_json_file(self, fname, dicts, file_type = 'w', encoding_type = 'utf-8') :
        with open(fname, file_type, encoding = encoding_type) as fp :
            json_data = json.dumps(dicts, ensure_ascii = False, indent = 4)
            fp.write(json_data)