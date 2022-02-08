import os
import argparse
from datetime import datetime
import torch
from model import build_model, build_backward_model
from model.solver import make_optimizer
from model.engine import do_train, do_evaluation, do_visualization
from data import make_data_loader
from config import get_cfg_defaults
from util.logger import setup_logger

import torch.nn as nn
import numpy as np

from model import conversion_helper as conversion

from model.transforms import *
CODINGS_HOME="../../age_estimation/encodings/"


import model.engine.pruning as pruning
def test(cfg, args, name):
    val_range = 700
    if args.transform == 'cnt':
        bits = 1 
        transform = None
        result_transform = RegressionTransform()
        criterion = nn.L1Loss()

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
    else:
        raise NotImplementedError
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
        
    model = build_model(cfg, bits)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # load last checkpoint
    if cfg.MODEL.WEIGHTS != "":
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))
    else:
        assert(False)

    # build the optimizer
    optimizer = make_optimizer(cfg, model)

    # build the dataloader
    dataloader_train = make_data_loader(cfg, 'train', transform)
    dataloader_val = make_data_loader(cfg, 'val', transform)
    # dataloader_val = None

    evaluation_with_model(cfg, model, criterion, result_transform, name, dataloader_val, device)


def train(cfg, args, name):
    val_range = 700
    if args.transform == 'cnt':
        bits = 1 
        transform = None
        result_transform = RegressionTransform()
        criterion = nn.L1Loss()

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
    else:
        raise NotImplementedError
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


    model = build_model(cfg, bits)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # load last checkpoint
    if cfg.MODEL.WEIGHTS != "":
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))

    # build the optimizer
    optimizer = make_optimizer(cfg, model)

    # build the dataloader
    dataloader_train = make_data_loader(cfg, 'train', transform)
    dataloader_val = make_data_loader(cfg, 'val', transform)
    # dataloader_val = None

    # start the training procedure
    do_train(
        cfg,
        args,
        model,
        dataloader_train,
        dataloader_val,
        optimizer,
        device,
        criterion,
        result_transform,
        name
    )

    evaluation_with_model(cfg, model, criterion, result_transform, name, dataloader_val, device)

    torch.save(model.state_dict(), name + ".pth.tar")


def evaluation_with_model(cfg, model, criterion, transform, name, dataloader, device, dataset='val'):
    # start the inferring procedure
    do_evaluation(
        cfg,
        model,
        dataloader,
        device,
        criterion,
        transform,
        name,
        verbose=True
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Self-driving Car Training and Inference.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="file",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--mode",
        default="train",
        metavar="mode",
        help="'train' or 'test'",
        type=str,
    )
    parser.add_argument(
        "--train-mode",
        default="train",
        metavar="train_mode",
        help="'train' or 'finetune'",
        type=str,
    )
    parser.add_argument(
        "--model-file",
        default="",
        metavar="model_file",
        help="path to model file",
        type=str,
    )
    parser.add_argument(
        "--sparsity",
        default=0,
        metavar="sparsity",
        help="amount to prune (0-1)",
        type=float,
    )
    parser.add_argument(
        "--name",
        default="",
        metavar="name",
        help="name of run",
        type=str,
    )
    parser.add_argument('--transform', help='transform in [cnt, bcd, gray, temp, 1hot, 2hot, 4hot, nby2hot]', type=str)
    parser.add_argument('--reverse-transform', help='reverse-transform in [sf, cor, ex]', type=str)
    parser.add_argument('--loss', help='loss in BCE, CE, MAE', type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # setup the logger
    if not os.path.isdir(cfg.OUTPUT.DIR):
        os.mkdir(cfg.OUTPUT.DIR)

    name = "{}_{}_{}".format(args.transform, args.reverse_transform, args.loss)
    logger = setup_logger("balad-mobile.train", cfg.OUTPUT.DIR,
                          name + '{0:%Y-%m-%d %H:%M:%S}_log'.format(datetime.now()))
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))
    if args.mode == 'test':
        test(cfg, args, name)

    else:
        # TRAIN
        if args.train_mode == "train":
            train(cfg, args, name)

        

    # Visualize
    # visualization(cfg)

    # evaluation(cfg)


if __name__ == "__main__":
    main()
