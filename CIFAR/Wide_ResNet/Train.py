from pathlib import Path
import tensorflow as tf
from tensorflow.keras import backend as K
from Model import make_resnet
from Dataset import get_dataset
from Top_K_Accuracy import custom_top_k_metrics
from Save import Recoder
from Checkpoint import load_checkpoint
from Utils import create_folder, save_model_info

class Trainer() :
    def __init__(self, **kwargs) :
        if kwargs["test"] :
            self.mode = "test"
        elif kwargs["plot"] :
            self.mode = "plot"
        else :
            self.mode = "train"
        result_folder = kwargs["result_folder"]

        if self.mode == "train" :
            self.initial_epoch = 1
            self.epochs = kwargs["epochs"]
            self.lr = kwargs["lr"]
            self.sgd_momentum = kwargs["sgd_momentum"]

        self.num_category = kwargs["num_category"]
        self.use_label_smoothing = kwargs["label_smoothing"]
        if (kwargs["num_layers"] - 4) % (3 * kwargs["deepening_factor"]) :
            raise Exception("({} - 4) % (3 * {}) != 0".format(kwargs["num_layers"], kwargs["deepening_factor"]))

        if self.mode != "plot" :
            self.ckpt_path = kwargs["ckpt_path"]
            self.ckpt_epoch = kwargs["ckpt_epoch"]

        log_folder, self.ckpt_folder = create_folder(result_folder)
        if self.mode == "train" :
            result_file = log_folder / "training_result.txt"
            simple_result_file = log_folder / "training_result_summary.csv"
            self.recoder = Recoder(result_file, simple_result_file)
        top_k_file = log_folder / "top_k_accuracy.csv"

        shortcut = "identity" if kwargs["zero_padding"] else "projection"
        self.resnet = make_resnet(self.num_category, kwargs["num_layers"], kwargs["bn_momentum"], kwargs["widening_factor"],
                                  kwargs["deepening_factor"], shortcut)

        if self.mode == "test" :
            _, _, self.test_dataset, _, _, self.num_test = get_dataset(kwargs["batch_size"], self.num_category, self.use_label_smoothing)
            self.top_k_accuracy = custom_top_k_metrics(self.resnet, self.test_dataset, top_k_file, self.num_category, 10, 
                                                       self.num_test, self.use_label_smoothing)
        elif self.mode == "train" :
            self.train_dataset, self.val_dataset, _, self.num_train, self.num_val, _ = get_dataset(kwargs["batch_size"], self.num_category, 
                                                                                                   self.use_label_smoothing)
            self.top_k_accuracy = custom_top_k_metrics(self.resnet, self.val_dataset, top_k_file, self.num_category, 10, 
                                                       self.num_val, self.use_label_smoothing)

        # kwargs 값 저장
        msg = ""
        for k, v in list(kwargs.items()) :
            msg += "{} = {}\n".format(k, v)
        msg += "new model checkpoint path = {}\n".format(self.ckpt_folder)
        with (log_folder / "model_settings.txt").open("w", encoding = "utf-8") as fp :
            fp.write(msg)

        save_model_info(self.resnet, log_folder)

    def start(self) :
        if self.mode == "test" :
            self.test()
        elif self.mode == "train" :
            self.train()

    def train(self) :
        self.opt = tf.keras.optimizers.SGD(lr = self.lr, momentum = self.sgd_momentum, nesterov = True)

        self.train_loss_metric = tf.keras.metrics.Mean(name = "train_loss")
        if self.use_label_smoothing :
            self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name = "train_acc")
            self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
            train_function = tf.function(self.forward, input_signature = [tf.TensorSpec((None, 32, 32, 3), tf.float32), 
                                                                          tf.TensorSpec((None, self.num_category), tf.float32)])
        else :
            self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name = "train_acc")
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
            train_function = tf.function(self.forward, input_signature = [tf.TensorSpec((None, 32, 32, 3), tf.float32), 
                                                                          tf.TensorSpec((None, 1), tf.int32)])

        ckpt = tf.train.Checkpoint(model = self.resnet, opt = self.opt)
        if self.ckpt_path is not None :
            fname, self.initial_epoch = load_checkpoint(Path(self.ckpt_path).resolve(), self.ckpt_epoch)
            print("\nCheckpoint File : {}\n".format(fname))
            ckpt.mapped = {"model" : self.resnet, "opt" : self.opt}
            ckpt.restore(fname)
            K.set_value(self.opt.lr, self.lr)

        progbar = tf.keras.utils.Progbar(target = self.num_train)
        for epoch in range(self.initial_epoch, self.initial_epoch + self.epochs) :
            self.train_loss_metric.reset_states()
            self.train_acc_metric.reset_states()

            self.recoder.set_start_train()
            for X, y in self.train_dataset :
                num_data = K.int_shape(y)[0]
                train_function(X, y)
                progbar.add(num_data)
            self.recoder.set_end_train()
            progbar.update(0)

            self.recoder.set_start_val()
            val_loss, val_acc = self.top_k_accuracy.evaluate(epoch)
            self.recoder.set_end_val()

            train_loss = self.train_loss_metric.result()
            train_acc = self.train_acc_metric.result()

            ckpt_prefix = self.ckpt_folder / "Epoch-{}_TLoss-{:.4f}_VLoss-{:.4f}".format(epoch, train_loss, val_loss)
            ckpt.save(file_prefix = ckpt_prefix)

            self.recoder.record(epoch, self.opt.get_config()["learning_rate"], train_acc.numpy(), train_loss.numpy(),
                                val_acc, val_loss.numpy())

    def test(self) :
        ckpt = tf.train.Checkpoint(model = self.resnet)
        fname, _ = load_checkpoint(Path(self.ckpt_path).resolve(), self.ckpt_epoch)
        print("\nCheckpoint File : {}\n".format(fname))

        # model만 불러옴
        ckpt.mapped = {"model" : self.resnet}
        ckpt.restore(fname).expect_partial()

        self.top_k_accuracy.evaluate("Test")

    def forward(self, inputs, labels) :
        with tf.GradientTape() as tape :
            logits = self.resnet(inputs, training = True)
            loss = self.loss_function(labels, logits)

        grads = tape.gradient(loss, self.resnet.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.resnet.trainable_variables))

        self.train_loss_metric.update_state(loss)
        self.train_acc_metric.update_state(labels, logits)