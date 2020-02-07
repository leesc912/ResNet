from __future__ import print_function
from argparse import ArgumentParser, RawTextHelpFormatter
from argparse_type_check import True_False_check
import os, sys

from Train import train
from plot_model import plot

sys.setrecursionlimit(10000)

parser = ArgumentParser(formatter_class = RawTextHelpFormatter)

parser.add_argument('-p', '--plot', type = True_False_check, default = False,
    help = "Model의 구조 및 parameter 개수만 확인\n" + "type : bool - default : False\n\n")
        
parser.add_argument('-c', '--num_categories', type = int, default = 10, choices = [10, 100],
    help = 'cifar-10 또는 cifar-100 중 선택\n' +
        'type : int - default : 10 - choices = [10, 100]\n\n')

parser.add_argument('-f', '--folder', type = str, default = None,
    help = "model의 결과를 저장할 folder 경로\n" + "type : str - default : None\n\n")

args                            = parser.parse_args()
kwargs                          = vars(args)

if kwargs['plot'] :
    plot("ResNet_plot", **kwargs)
else :
    train(**kwargs)