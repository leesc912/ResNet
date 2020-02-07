import os, sys
from Train import train

sys.setrecursionlimit(10000)

kwargs                                  = dict()
kwargs['epochs']                        = 100000
kwargs['start_epochs']                  = 0         # None or numeric value
kwargs['batch_size']                    = 128       
kwargs['base_lr']                       = 0.01      # None or numeric value
kwargs['val_split_ratio']               = 0.2
kwargs['verbose']                       = 1

kwargs['num_filters']                   = [16, 32, 64]
kwargs['kernel_size']                   = 3
kwargs['kernel_initializer']            = 'glorot_uniform'
kwargs['l2_value']                      = 0.0001
kwargs['norm_momentum']                 = 0.9
kwargs['num_box']                       = 3
kwargs['num_blocks_in_box']             = [18, 18, 18]
kwargs['num_categories']                = 10
kwargs['opt_type']                      = "SGD"
if kwargs['opt_type'] == 'SGD' :
    kwargs['SGD_momentum']              = 0.9

kwargs['bottleneck']                    = False
kwargs['pre_activation']                = False
kwargs['zero_pad']                      = True
kwargs['data_augmentation']             = True

lr_scheduler                            = "SLRC"

# SpecificLearningRateScheduler                     
if lr_scheduler == "SLRC" : 
    kwargs['lr_scheduler']              = "SLRC"
    kwargs['decay_step_list']           = [400, 32000, 48000]
    kwargs['lr_list']                   = [0.1, 0.01, 0.001]
    
# ConditionalLearningRateScheduler
elif lr_scheduler == "CLRS" : 
    kwargs["lr_scheduler"]              = "CLRS"
    kwargs['lr_decay']                  = 0.5
    kwargs['min_lr']                    = 0.0001
    kwargs['patience']                  = 10
    kwargs['history_type']              = 'val_loss'
    kwargs['threshold_acc']             = 0
    kwargs['num_previous_history']      = 0
    kwargs['continue_when_min_lr']      = False

else :
    kwargs['lr_scheduler']              = None

kwargs['print_in_top_k_acc']            = True
if kwargs['print_in_top_k_acc'] :
    kwargs['max_in_top_k']              = 10


'''
    kwargs['ckpt_model_path']는 Train.py에서의 save_folder 값 또는 [*.json, *.h5] 형식의 list type
    list일 때, *.json과 *.h5의 순서는 상관없음.

    ex) kwargs['ckpt_model_path'] = os.path.join(".", "cifar_result", "2019-04-18__004330") - kwargs['load_type']이 필요함
    ex) kwargs['ckpt_model_path'] = [".../model_weights.h5", ".../model_structure.json"]

    assert kwargs['load_type'] in ['min_val_loss_ckpt', 'max_val_acc_ckpt', 'the_latest_ckpt', specific epoch(type : int)] 
'''

kwargs['ckpt_model_path']               = None  
if kwargs['ckpt_model_path'] is not None :
    kwargs['load_type']                 = 'min_val_loss_ckpt'

train(**kwargs)