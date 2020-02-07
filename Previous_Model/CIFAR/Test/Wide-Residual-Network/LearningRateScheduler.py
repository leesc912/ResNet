import tensorflow as tf
import tensorflow.keras.backend as K

def change_lr(model, new_lr, min_lr = None) :
    previous_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, new_lr)

    print("\nOptimizer Learning Rate [{:10.8f}] -> [{:10.8f}] (minimum lr : {})\n".format(
        previous_lr, K.get_value(model.optimizer.lr), min_lr))

class SpecificLearningRateScheduler(tf.keras.callbacks.Callback) :
    # 정해진 epoch에 미리 정의한 learning rate로 바꿈
    def __init__(self, epoch) :
        super(SpecificLearningRateScheduler, self).__init__()

        self.decay_step_list = [60, 120, 160]
        self.lr_list = 0.2
        self.cur_idx = 0
        self.step_length = len(self.decay_step_list)
        self.check_condition()

        # initial epoch에 맞춰서 current index를 설정
        while self.cur_idx < self.step_length and self.decay_step_list[self.cur_idx] < epoch :
            self.cur_idx += 1

    def on_epoch_begin(self, epoch, logs = {}) :
        if self.cur_idx < self.step_length and epoch == self.decay_step_list[self.cur_idx] :
            change_lr(
                model = self.model, 
                new_lr = (self.lr_list[self.cur_idx] if isinstance(self.lr_list, list) 
                            else self.model.optimizer.lr * self.lr_list),
                min_lr = None
            )
            self.cur_idx += 1

    def check_condition(self) :
        self.decay_step_list = list(self.decay_step_list)
        if not isinstance(self.lr_list, float) :
            self.lr_list = list(self.lr_list)
        
        if len(self.decay_step_list) > 1 and isinstance(self.lr_list, list) :
            if len(self.lr_list) > 1 :
                assert len(self.decay_step_list) == len(self.lr_list)
                assert all(e1 > 0 and e2 > 0 for e1, e2 in zip(self.decay_step_list, self.lr_list))
            else :
                self.lr_list = float(self.lr_list[0])