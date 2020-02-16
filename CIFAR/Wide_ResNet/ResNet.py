from __future__ import print_function
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from type_check import bool_type_check
from Train import Trainer

sys.setrecursionlimit(10000)

parser = ArgumentParser(formatter_class = RawTextHelpFormatter)

parser.add_argument('-t', '--test', type = bool_type_check, default = False,
                    help = "\n테스트 모드. 테스트를 위한 checkpoint를 전달하지 않으면 error 발생\n" + "default : False\n\n")

parser.add_argument('-p', '--plot', type = bool_type_check, default = False,
                    help = "\nModel의 구조 및 parameter 개수만 확인\n" + "Priority : test > plot > train\n" + 
                           "default : False\n\n")

parser.add_argument('-f', '--result_folder', type = str, default = None,
                    help = "\n모델의 진행 사항을 저장할 폴더\n" + "default : 현재 위치에 result folder 생성\n\n")

parser.add_argument('-e', '--epochs', type = int, default = 200, 
                    help = "\ndefault - 200\n\n")

parser.add_argument('-b', '--batch_size', type = int, default = 128, 
                    help = "\ndefault : 128\n\n")

parser.add_argument('-r', '--lr', type = float, default = 0.1,
                    help = "\nlearning rate\n" + "default : 0.1\n\n")

parser.add_argument('-n', '--num_category', type = int, default = 10, choices = [10, 100],
                    help = "\ncifar-10 또는 cifar-100 중 선택\n" + "default : 10\n\n")

parser.add_argument('-N', '--num_layers', type = int, default = 28,
                    help = r"\nResNet-{num_layers}' + '\n" + "(num_layers - 4)가 6의 배수가 아니라면 Error 발생\n" +
                            "default : 28\n\n")

parser.add_argument('-w', '--widening_factor', type = int, default = 10, help = "\nWRN의 k\n" + "default : 10\n\n")

parser.add_argument('-L', '--deepening_factor', type = int, default = 2, help = "\nWRN의 l\n" + "default : 2\n\n")

parser.add_argument('-z', '--zero_padding', type = bool_type_check, default = False,
                    help = "\nshortcut의 dimensions을 늘릴 때 Conv2D 대신 tf.pad 사용\n" + "default : False\n\n")

parser.add_argument('-m', '--sgd_momentum', type = float, default = 0.9,
                    help = "\nSGD Optimizer의 moemntum\n" + "default : 0.9\n\n")

parser.add_argument('-M', '--bn_momentum', type = float, default = 0.9,
                    help = "\nBatchNormalization layer의 momentum\n" + "default : 0.9\n\n")

parser.add_argument('-l', '--label_smoothing', type = bool_type_check, default = False,
                    help = "\nLabel Smoothing 사용\n" + "default : False\n\n")

parser.add_argument('-P', '--ckpt_path', type = str, default = None,
                    help = "\ncheckpoint path - default : None\n" + 
                           "argument는 Train.py에서 folder 값 또는 checkpoint file name\n" +
                           "ex1) -c ./foo/results/2019-04-18__004330\n" +
                           "ex2) -c ./foo/results/2019-04-18__004330/ckpt.file\n\n")

parser.add_argument('-E', '--ckpt_epoch', type = int, default = None,
                    help = "\ncheckpoint path가 folder일 경우 불러올 checkpoint의 epoch\n" +
                           "만약 checkpoint의 path가 folder일 때, checkpoint_epoch를 설정하지 않으면\n" +
                           "가장 최근의 checkpoint를 불러옴\n\n")

args                        = parser.parse_args()
kwargs                      = vars(args)

model = Trainer(**kwargs)
model.start()