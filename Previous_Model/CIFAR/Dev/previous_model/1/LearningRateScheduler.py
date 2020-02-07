import tensorflow as tf
import tensorflow.keras.backend as K

def change_lr(model, new_lr, min_lr = None) :
    previous_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, new_lr)
    print("\nOptimizer Learning Rate [{:10.8f}] -> [{:10.8f}] (minimum lr : {})\n".format(
        previous_lr, K.get_value(model.optimizer.lr), min_lr))

class SpecificLearningRateScheduler(tf.keras.callbacks.Callback) :
    # 정해진 epoch에 미리 정의한 learning rate로 바꿈
    def __init__(self, *args, **kwargs) :
        super(SpecificLearningRateScheduler, self).__init__()

        self.decay_step_list = kwargs["decay_step_list"]
        self.lr_list = kwargs["lr_list"]
        self.cur_idx = 0
        self.step_length = len(self.decay_step_list)
        self.check_condition()

    def on_epoch_begin(self, epoch, logs = {}) :
        if self.cur_idx < self.step_length and epoch == self.decay_step_list[self.cur_idx] :
            change_lr(model = self.model, new_lr = self.lr_list[self.cur_idx], min_lr = None)
            self.cur_idx += 1

    def check_condition(self) :
        assert isinstance(self.decay_step_list, list) or isinstance(self.decay_step_list, tuple)
        assert isinstance(self.lr_list, list) or isinstance(self.lr_list, tuple)
        assert len(self.decay_step_list) == len(self.lr_list)
        assert all(e1 > 0 and e2 > 0 for e1, e2 in zip(self.decay_step_list, self.lr_list))

class ConditionalLearningRateScheduler(tf.keras.callbacks.Callback) :
    # 일정 시간 loss나 accuracy가 변화가 없을 때 learning rate를 바꿈
    def __init__(self, *args, **kwargs) : 
        super(ConditionalLearningRateScheduler, self).__init__()
        
        self.lr_decay = kwargs["lr_decay"]
        self.patience = kwargs["patience"]
        self.min_lr = kwargs["min_lr"]
        self.continue_when_min_lr = kwargs["continue_when_min_lr"]
        self.threshold_acc = kwargs["threshold_acc"]
        self.status = True if self.threshold_acc == 0 else False

        self.num_previous_history = kwargs["num_previous_history"]
        self.history_type = 'val_acc' if kwargs["history_type"] == 'val_acc' else 'val_loss'
        self.historyList = {'tr_acc' : [], 'tr_loss' : [], 'val_acc' : [], 'val_loss' : []}

        self.count = 0

        self.check_condition()

    def on_epoch_end(self, epoch, logs = {}) :
        t_acc = logs['acc']
        t_loss = logs['loss']
        val_acc = logs['val_acc']
        val_loss = logs['val_loss']

        self.historyList['tr_acc'].append(t_acc)
        self.historyList['tr_loss'].append(t_loss)
        self.historyList['val_acc'].append(val_acc)
        self.historyList['val_loss'].append(val_loss)

        if len(self.historyList[self.history_type]) > 1 :
            history_list = self.historyList[self.history_type][ : -1]
            if self.num_previous_history :
                history_list = history_list[-self.num_previous_history -1 : ]

            if self.history_type == 'val_loss' :
                value = min(history_list)
                if value < val_loss :
                    self.count += 1
                else :
                    self.count = 0
            else : # val_acc
                value = max(history_list)
                if value > val_acc :
                    self.count += 1
                else :
                    self.count = 0

        if self.status :
            if self.count > self.patience :
                lr = K.get_value(self.model.optimizer.lr)
                if (lr <= self.min_lr) and (not self.continue_when_min_lr) :
                    self.model.stop_training = True
                else :
                    new_lr = lr * self.lr_decay
                    change_lr(model = self.model, new_lr = new_lr, min_lr = self.min_lr)

                self.count = 0
        else :
            if t_acc > self.threshold_acc :
                self.status = True
            self.count = 0 # count 변수 필요없음

    def check_condition(self) :
        assert self.lr_decay > 0
        assert self.patience > 0
        assert self.num_previous_history >= 0                
        assert self.threshold_acc >= 0
        assert self.min_lr > 0
        assert isinstance(self.continue_when_min_lr, bool)