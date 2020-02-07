from argparse import ArgumentTypeError

def initial_lr_type_check(string) :
    status = False

    if string == 'None' :
        return None
    else :
        try :
            value = float(string)
            if float(string) != int(string) :
                status = False

        except ValueError :
            status = False

    if status :
        return value
    else :
        raise ArgumentTypeError(
            "\n\tinitial_lr의 type은 None 또는 float (입력된 initial_lr : {})\n".format(string)
        )

def load_type_check(string) :
    if string not in ['min_loss_ckpt', 'max_acc_ckpt', 'the_latest_ckpt'] :
        try :
            value = int(string)
            return value
        except ValueError :
            raise ArgumentTypeError(
                "\n\tload_type은 'min_loss_ckpt', 'max_acc_ckpt', 'the_latest_ckpt' " + 
                "또는 integer (입력된 load_type : {})\n".format(string)
            )
    else :
        return string

def initial_epoch_type_check(string) :
    status = True

    if string in ['0', 'None'] :
        return None
    else :
        try :
            value = int(string)
        except ValueError :
            status = False

    if status :
        return value
    else :
        raise ArgumentTypeError(
            "\n\tinitial_epoch의 type은 None 또는 int (입력된 initial_epoch : {})\n".format(string)
        )

def True_False_check(string) :
    if string.lower() == 'true' :
        return True
    elif string.lower() == 'false' :
        return False
    else :
        raise ArgumentTypeError(
            "\n\tTrue 또는 False만 입력 가능 (입력된 값 : {})\n".format(string)
        )