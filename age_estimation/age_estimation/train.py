# from age_estimation.ageresnet.data.transforms import e2jmjReverseTransform
import argparse

from train_main_iw import train_main as train_main_iw
from train_main_afad import train_main as train_main_afad
from test_main_afad import test_main as test_main_afad
from train_main_morph2 import train_main as train_main_morph2
from test_main_morph2 import test_main as test_main_morph2

from train_main_afad_thin import train_main as train_main_afad_thin
from train_main_morph2_thin import train_main as train_main_morph2_thin


import random
from ageresnet import *

import math

def closest_power2(x):
    """
    Return the closest power of 2 by checking whether 
    the second binary number is a 1.
    """
    op = math.floor if bin(x)[3] != "1" else math.ceil
    return 2**(op(math.log(x,2)))

def parse_args():
    parser = argparse.ArgumentParser(description='Train age estimation')
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--model', help='model', type=str, default='resnet')
    parser.add_argument('--num-epochs', help='epochs', type=int)
    parser.add_argument('--sparsity', help='sparsity', type=float, default=0)
    parser.add_argument('--label', help='identifying label', type=str)
    parser.add_argument('--dataset', help='dataset in [iw, afad, morph2]', type=str)
    parser.add_argument('--transform', help='transform in [cnt, bcd, gray, temp, 1hot, 2hot, 4hot, nby2hot]', type=str)
    parser.add_argument('--reverse-transform', help='reverse-transform in [sf, cor, ex]', type=str)
    parser.add_argument('--loss', help='loss in BCE, CE, MAE', type=str)
    parser.add_argument('--ckpt', help='ckpt file', type=str)
    parser.add_argument('--mode', help='mode in [train, test]', type=str, default='train')
    parser.add_argument('--data-aug', help='use data aug from DLDL-v2', action='store_true')

    args = parser.parse_args()
    return args

CODINGS_HOME="../encodings/"

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'iw':
        val_range = 105 # 0-101 + more?
    elif args.dataset == 'afad':
        val_range = 26 # 15 - 40
    elif args.dataset == 'morph2':
        val_range = 62 # 16 - 77

    if args.transform == 'cnt':
        bits = 1 
        transform = None
        result_transform = RegressionTransform()
        criterion = AgeL1Loss()

    elif args.transform == 'u':
        transform = FileTransform(CODINGS_HOME + "belu_" + str(val_range) + "_tensor.pkl")  
        result_transform = TempReverseTransform(val_range)

    elif args.transform == 'j':
        transform = FileTransform(CODINGS_HOME + "belj_" + str(val_range) + "_tensor.pkl")
        result_transform = nby2ReverseTransform(val_range)

    elif args.transform == 'h':
        transform = FileTransform(CODINGS_HOME + "hex16_" + str(val_range) + "_tensor.pkl")
        
    elif args.transform == 'e1':
        transform = FileTransform(CODINGS_HOME + "belb1jdj_" + str(val_range) + "_tensor.pkl")

    elif args.transform == 'e2':
        transform = FileTransform(CODINGS_HOME + "belb2jdj_" + str(val_range) + "_tensor.pkl")  

    elif args.transform == 'had':
        transform = FileTransform(CODINGS_HOME + "had_" + str(val_range) + "_tensor.pkl")  
    
    elif args.transform == 'mc':
        transform = FileTransform(CODINGS_HOME + "belu_" + str(val_range) + "_tensor.pkl")  
    bits = transform.bits
    
    if args.reverse_transform == 'cor':
        result_transform = correlationReverseTransform(transform)
    elif args.reverse_transform == 'ex':
        result_transform = softCorrelationReverseTransform(transform)
    elif args.reverse_transform == 'exn':
        result_transform = softCorrelationReverseTransform(transform)
    elif args.reverse_transform == 'mc':
        result_transform = MultiClassReverseTransform()
    elif args.reverse_transform != 'sf':
        raise NotImplementedError

    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss(reduction="sum").cuda()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(reduction="sum").cuda()
    elif args.loss == 'mse':
        criterion = nn.MSELoss(reduction="mean").cuda()
    elif args.loss == 'mae':
        criterion = nn.L1Loss(reduction="mean").cuda()
    
    # for i in range(val_range):
    #     enc = transform(i)
    #     for j in range(transform.bits):
    #         if enc[j] == 0:
    #             enc[j] = -1
    #     enc = enc.unsqueeze(0).cuda()
    #     dec = result_transform(enc)
    #     # dec_2 = result_transform.convert_discrete(enc)
    #     # dec_3 = result_transform.convert_continuous(enc)
    #     print(i, dec)
    # exit()
    # if args.reverse_transform == "ex":
    #     exit()

    # smaller networks
    if args.dataset == 'iw':
        if args.mode == 'traintan':
            train_main_iw(args, transform, result_transform, criterion, bits)
    elif args.dataset == 'afad':
        if args.mode == 'traintan':
            train_main_afad_thin(args, transform, result_transform, criterion, bits)
    elif args.dataset == 'morph2':
        if args.mode == 'traintan':
            train_main_morph2_thin(args, transform, result_transform, criterion, bits)

    
    if args.dataset == 'iw':
        if args.mode == 'train':
            train_main_iw(args, transform, result_transform, criterion, bits)
    elif args.dataset == 'afad':
        if args.mode == 'train':
            train_main_afad(args, transform, result_transform, criterion, bits)
        else:
            test_main_afad(args, transform, result_transform, criterion, bits)

    elif args.dataset == 'morph2':
        if args.mode == 'train':
            train_main_morph2(args, transform, result_transform, criterion, bits)
        else:
            test_main_morph2(args, transform, result_transform, criterion, bits)


