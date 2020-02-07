import os
import glob
import re
import json
import tensorflow as tf

def save_model_info(model, save_folder, model_structure_path, model_weights_path, kwargs) :
    initial_log_file = os.path.join(save_folder, "initial_log.txt")

    with open(initial_log_file, "a+", encoding = 'utf-8') as fp :
        strList = ["{} = {}".format(keyword, value) for keyword, value in kwargs.items()]
        strList.append("{} = {}".format("new_model_structure_path", model_structure_path))
        strList.append("{} = {}".format("new_model_weights_path", model_weights_path)) 
        strList.append("\n" * 3)
        fp.write("\n".join(strList))
            
    tf.keras.utils.plot_model(model, to_file = os.path.join(save_folder, "model.png"))
    model_json = model.to_json()
    with open(os.path.join(model_structure_path, "model_structure.json"), "w", encoding = 'utf-8') as json_file :
        json_file.write(model_json)
    model.save_weights(filepath = os.path.join(model_weights_path, "init-ckpt.h5"))

    # summary 저장
    strList = []
    model.summary(print_fn=lambda x: strList.append(x))
    model_summary = "\n".join(strList)
    
    with open(initial_log_file, "a+", encoding = 'utf-8') as fp :
        fp.write(model_summary)

def extract_epoch_and_lr(weights_path) :
    fname = os.path.split(weights_path)[-1]

    epoch_str = re.findall("epoch-\d+", fname)
    lr_str = re.findall("LR-\d+.\d+", fname)

    start_epochs = int(epoch_str[0].split("-")[-1]) + 1 if len(epoch_str) else None
    lr = float(lr_str[0].split("-")[-1]) if len(lr_str) else None
        
    return start_epochs, lr

def load_checkpoint(*args, **kwargs) :
    ckpt_model_path = kwargs["ckpt_model_path"]
    load_type = kwargs["load_type"]

    if not isinstance(ckpt_model_path, list) :
        assert os.path.exists(ckpt_model_path)
        assert load_type in ['min_val_loss_ckpt', 'max_val_acc_ckpt', 'the_latest_ckpt'] or isinstance(load_type, int)

        print("\nload_type : {}".format(load_type))
        ckpt_model_structure_path = os.path.join(ckpt_model_path, "model", "structure", "model_structure.json")
        ckpt_model_weights_path = os.path.join(ckpt_model_path, "model", "weights")

        ckpt_info_file = os.path.join(ckpt_model_weights_path, "model-ckpt-info.json")

        with open(ckpt_info_file, "r", encoding = 'utf-8') as json_file :
            ckpt_info = json.loads(json_file.read())
        
        weights_file = None
        if not isinstance(load_type, int) :
            weights_file = os.path.join(ckpt_model_weights_path, ckpt_info[load_type]['path'])
        else :
            epoch_str = 'epoch-{}'.format(load_type)
            weights_file_list = glob.glob(os.path.join(ckpt_model_weights_path, "*.h5"))
            for _file in weights_file_list :
                if _file.startswith(os.path.join(ckpt_model_weights_path, epoch_str)) :
                    weights_file = _file
                    break

        start_epochs, lr = extract_epoch_and_lr(weights_file)

    elif isinstance(ckpt_model_path, list) :
        if os.path.splitext(ckpt_model_path[0])[-1] == ".json" :
            ckpt_model_structure_path = ckpt_model_path[0]
            weights_file = ckpt_model_path[1]
        else :
            ckpt_model_structure_path = ckpt_model_path[1]
            weights_file = ckpt_model_path[0]

        start_epochs, lr = extract_epoch_and_lr(weights_file)

    else :
        raise Exception()

    print("ckpt_weights_path : {}".format(weights_file))
    print("ckpt_structure_path : {}".format(ckpt_model_structure_path))
    print("ckpt learning rate : {}".format(lr))
    print("ckpt start_epochs : {}\n".format(start_epochs))

    with open(ckpt_model_structure_path, "r", encoding = 'utf-8') as json_file :
        model = tf.keras.models.model_from_json(json_file.read())
    model.load_weights(weights_file)

    return model, lr, start_epochs