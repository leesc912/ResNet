from __future__ import print_function
from argparse import ArgumentParser, RawTextHelpFormatter
from argparse_type_check import (
    initial_lr_type_check, initial_epoch_type_check, load_type_check, True_False_check)
import os, sys

from Train import train
from Keras_utils.plot_model import plot

sys.setrecursionlimit(10000)

parser = ArgumentParser(formatter_class = RawTextHelpFormatter)

parser.add_argument('-p', '--plot', type = True_False_check, default = False,
    help = "Model의 구조 및 parameter 개수만 확인\n" + "type : bool - default : False\n\n")

parser.add_argument('-e', '--epochs', type = int, default = 100000, 
    help = "0 이라면 Training 대신 Test를 수행\n" + "type : int - default - 100000\n\n")

parser.add_argument('-E', '--initial_epoch', type = initial_epoch_type_check, default = None,
    help = "초기 epoch 값\n" + "default : None\n\n")

parser.add_argument('-l', '--initial_lr', type = initial_lr_type_check, default = 0.01,
    help = '초기 learning rate\n' + 'type : None or float - default : 0.01\n\n')

parser.add_argument('-k', '--kernel_initializer', type = str, default = 'glorot_uniform',
    help = "Conv2D, Dense layer의 kernel initializer\n" +
        "type : str - default : 'glorot_uniform'\n\n")

parser.add_argument('-m', '--momentum', type = float, default = 0.9,
    help = 'batch normalization의 momentum 값\n' + 'type : float - default : 0.9\n\n')

parser.add_argument('-c', '--num_categories', type = int, default = 10, choices = [10, 100],
    help = 'cifar-10 또는 cifar-100 중 선택\n' +
        'type : int - default : 10 - choices = [10, 100]\n\n')

parser.add_argument('-n', '--num_layers', type = int, default = 110,
    help = r'ResNet-{num_layers}' + '\n' +
        '(num_layers - 2)가 6의 배수가 아니라면 Error 발생\n' +
        'type : int - default : 110\n\n')

parser.add_argument('-z', '--zero_pad', type = True_False_check, default = True,
    help = 'shortcut의 dimensions을 늘릴 때 Conv2D 대신 tf.pad 사용\n' +
        'type : bool - default : True\n\n')

parser.add_argument('-s', '--lr_scheduler', type = True_False_check, default = False,
    help = 'learning rate scheduler 사용\n' + 'type : bool - default : False\n\n')

parser.add_argument('-d', '--decay_step_list', nargs = '+', type = int, default = None,
    help = 'learning rate에 변화를 줄 epoch\n' +
        '\n각 value는 learning rate에 변화가 일어나는 epoch\n' +
        'ex) "400, 32000, 48000" <- 400, 32000, 48000 epoch마다 learning rate 변경\n' +
        '\nnargs : "+" - type : int - default : None\n\n')

parser.add_argument('-L', '--lr_list', nargs = '+', type = float, default = None,
    help = 'list type이며 각 value는 변화될 learning rate를 나타냄\n' +
        'ex) "0.1, 0.01, 0.001" <- 특정 epoch마다 learning rate를 0.1, 0.01, 0.001로 변경\n' +
        '\nnargs : "+" - type : float - default : None\n\n')

parser.add_argument('-f', '--base_folder', type = str, default = None,
    help = "model의 결과를 저장할 folder 경로\n" + "type : str - default : None\n\n")

parser.add_argument('-C', '--ckpt_model_path', nargs = '+', type = str, default = None,
    help = 'checkpoint가 저장된 folder 또는 file 경로\n' +
    help = 'checkpoint를 불러올 때 필요한 argument\n' + 
        'ex1) --ckpt_model_path /home/lee/ckpt/2019-04-18__004330\n' +
        'ex2) --ckpt_model_path ./ckpt/model_structure.json ./ckpt/model_weights.h5\n' +
        'list를 전달할 때는 --load_type을 설정할 필요가 없음\n' +
        '\ntype : str - default : None\n\n')

parser.add_argument('-t', '--load_type', type = load_type_check, default = None,
    help = '불러올 checkpoint의 epoch\n' +
        '\n"min_loss_ckpt" : validation data에 대한 loss가 가장 작은 checkpoint를 불러옴\n' +
        '"max_acc_ckpt" : validation data에 대한 acc가 가장 큰 checkpoint를 불러옴\n' + 
        '"the_latest_ckpt" : 해당 폴더에서 가장 최근의 checkpoint를 불러옴\n' +
        '"specific epoch" : 특정 epoch의 checkpoint를 불러옴 (type : int)\n' +
        '\ndefault : None\n\n')


args                        = parser.parse_args()
kwargs                      = vars(args)
kwargs['model_structure']   = 'original'

if kwargs['plot'] :
    plot("ResNet_plot", **kwargs)
else :
    train(**kwargs)