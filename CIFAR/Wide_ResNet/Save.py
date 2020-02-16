import csv
from Utils import korea_time

class Recoder() :
    def __init__(self, result_file, simple_result_file) :
        self.result_file = result_file
        self.simple_result_file = simple_result_file

        self.start_train = None
        self.end_train = None
        self.start_val = None
        self.end_val = None

        # validation에 대한 결과만 저장
        self.good_val_result = {"min_loss" : {"epoch" : 0, "value" : None}, "max_acc" : {"epoch" : 0, "value" : None}}
        self.good_train_result = {"min_loss" : {"epoch" : 0, "value" : None}, "max_acc" : {"epoch" : 0, "value" : None}}

        self.csv_header = ["Epoch", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"]
        with self.simple_result_file.open("w", newline = '', encoding = "utf-8") as fp :
            writer = csv.DictWriter(fp, fieldnames = self.csv_header)
            writer.writeheader()

    def set_start_train(self) :
        self.start_train = korea_time(None)
    
    def set_end_train(self) :
        self.end_train = korea_time(None)

    def set_start_val(self) :
        self.start_val = korea_time(None)

    def set_end_val(self) :
        self.end_val = korea_time(None)

    def record(self, epoch, lr, train_acc, train_loss, val_acc, val_loss) :
        self.update_dic(self.good_train_result, epoch, train_acc, train_loss)
        self.update_dic(self.good_val_result, epoch, val_acc, val_loss)

        with self.result_file.open("a+", encoding = "utf-8") as fp :
            msg = "Epoch = [{:5d}]\n".format(epoch)
            msg += "Learning Rate    | [{:.6f}]\n".format(lr)
            msg += "Time             | Start = [ {} ]   End = [ {} ]\n".format(str(self.start_train.strftime("%Y-%m-%d %H:%M:%S")), 
                                                                               str(self.end_val.strftime("%Y-%m-%d %H:%M:%S")))
            msg += "Elapsed Time     | Train = [ {} ]        Val = [ {}] \n\n".format(self.end_train - self.start_train, self.end_val - self.start_val)
            msg += "Result   (Train) | Accuracy = [{:8.6f}]   Loss = [{:8.6f}]\n".format(train_acc, train_loss)
            msg += "Result   (Val)   | Accuracy = [{:8.6f}]   Loss = [{:8.6f}]\n\n".format(val_acc, val_loss)
            msg += "Min Loss (Train) | Epoch = [{:5d}]   Value = [{:8.6f}]\n".format(*self.get_result(self.good_train_result, "loss"))
            msg += "Max Acc  (Train) | Epoch = [{:5d}]   Value = [{:8.6f}]\n".format(*self.get_result(self.good_train_result, "acc"))
            msg += "Min Loss (Val)   | Epoch = [{:5d}]   Value = [{:8.6f}]\n".format(*self.get_result(self.good_val_result, "loss"))
            msg += "Max ACC  (Val)   | Epoch = [{:5d}]   Value = [{:8.6f}]\n\n".format(*self.get_result(self.good_val_result, "acc"))
            msg += " - " * 15 + "\n\n"
            fp.write(msg)

        with self.simple_result_file.open("a+", newline = '', encoding = "utf-8") as fp :
            writer = csv.DictWriter(fp, fieldnames = self.csv_header)
            row_dic = {"Epoch" : epoch, "Training Loss" : round(train_loss, 4), "Training Accuracy" : round(train_acc, 4), 
                       "Validation Loss" : round(val_loss, 4), "Validation Accuracy" : round(val_acc, 4)}
            writer.writerow(row_dic)
            
    def get_result(self, dic, mode) :
        if mode == "loss" :
            return dic["min_loss"]["epoch"], dic["min_loss"]["value"]
        else :
            return dic["max_acc"]["epoch"], dic["max_acc"]["value"]

    def update_dic(self, dic, epoch, acc, loss) :
        if dic["min_loss"]["epoch"] == 0 :
            dic["min_loss"]["epoch"] = epoch
            dic["min_loss"]["value"] = loss
        else :
            if dic["min_loss"]["value"] > loss :
                dic["min_loss"]["epoch"] = epoch
                dic["min_loss"]["value"] = loss

        if dic["max_acc"]["epoch"] == 0 :
            dic["max_acc"]["epoch"] = epoch
            dic["max_acc"]["value"] = acc
        else :
            if dic["max_acc"]["value"] < acc :
                dic["max_acc"]["epoch"] = epoch
                dic["max_acc"]["value"] = acc