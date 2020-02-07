import os
import glob
import re
import json
import tensorflow as tf

def save_model_info(model, save_folder, model_structure_path, model_weights_path, **kwargs) :
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