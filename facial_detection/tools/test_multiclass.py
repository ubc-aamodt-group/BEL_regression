import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config, update_config_code
from lib.datasets import get_dataset
from lib.core import function_classification as function
from lib.utils import utils
from torchsummary import summary
from torchvision import transforms
import torchvision
import torch.utils.model_zoo as model_zoo
import pruning
import os
import shutil
from ptflops import get_model_complexity_info
from pathlib import Path

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--code', help='code configuration filename',
                        required=True, type=str)
    parser.add_argument('--suf', help='suffix for logfile',
                        required=True, type=str)
    parser.add_argument('--model_file', help='model_file',
                        required=True, type=str)
    parser.add_argument('--loss', help='loss- bce/mse/ce/mse_tanh/ce_tanh',
                        required=True, type=str)
    args = parser.parse_args()
    update_config(config, args)
    update_config_code(config, args)
    return args


def main():

    args = parse_args()
    suf = args.suf+config.CODE.CODE_NAME+"_multiclass_"+args.loss
    loss_function=args.loss
    model_file=args.model_file
    root_output_dir = Path(config.OUTPUT_DIR)

    dataset = config.DATASET.DATASET
    model = config.MODEL.NAME
    cfg_name = (os.path.basename(args.cfg).split('.')[0])+str(config.TRAIN.LR)+str(config.SUF)+str(suf)

    final_output_dir = root_output_dir / dataset / cfg_name


    logger, final_new_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test',suf)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net6(config)

    #macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
    #                                       print_per_layer_stat=True, verbose=True)
    #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    gpus = list(config.GPUS)
    model = model.cuda(gpus[0])
    #print(model)
    #model_file = os.path.join(final_output_dir,'model_best.pth')
    #model_file = os.path.join(final_output_dir,'checkpoint_0.pth')
    #print(model_file)
    #if True:#os.path.islink(model_file): 
    #    print("exists")
    #    state_dict = torch.load(model_file)
    #    #print(state_dict.keys())
    #    #model.module.load_state_dict(state_dict)
    #    #if 'state_dict' in state_dict.keys():
    #    #    print("state_dir")
    #    #state_dict = state_dict['state_dict']
    #    model.load_state_dict(state_dict)
    #    #else:
    #    #    model.module.load_state_dict(state_dict)
    #else: 
    #    print("The model file %s does not exist"%(model_file))
    #    sys.exit()
    #model.load_state_dict(torch.load("temp.pth"))
    #model.load_state_dict(torch.load(final_output_dir+"/model_best.pth"))
    model.load_state_dict(torch.load(model_file))
    model.eval()
    dataset_type = get_dataset(config)

    test_file= config.DATASET.TESTSET
    val_loader = DataLoader(
        dataset=dataset_type(config,test_file=test_file,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    if config.DATASET.DATASET=="WFLW":
        test_file="./data/wflw/face_landmarks_wflw_test_blur.csv"
        blur_loader = DataLoader(
            dataset=dataset_type(config,test_file=test_file,
                                is_train=False),
            batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        test_file="./data/wflw/face_landmarks_wflw_test_expression.csv"
        expression_loader = DataLoader(
            dataset=dataset_type(config,test_file=test_file,
                                is_train=False),
            batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        test_file="./data/wflw/face_landmarks_wflw_test_illumination.csv"
        ill_loader = DataLoader(
            dataset=dataset_type(config,test_file=test_file,
                                is_train=False),
            batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        test_file="./data/wflw/face_landmarks_wflw_test_largepose.csv"
        large_loader = DataLoader(
            dataset=dataset_type(config,test_file=test_file,
                                is_train=False),
            batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        test_file="./data/wflw/face_landmarks_wflw_test_makeup.csv"
        make_loader = DataLoader(
            dataset=dataset_type(config,test_file=test_file,
                                is_train=False),
            batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        test_file="./data/wflw/face_landmarks_wflw_test_occlusion.csv"
        occ_loader = DataLoader(
            dataset=dataset_type(config,test_file=test_file,
                                is_train=False),
            batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )
    elif config.DATASET.DATASET=="AFLW":
        test_file="./data/aflw/face_landmarks_aflw_test_frontal.csv"
        front_loader = DataLoader(
        dataset=dataset_type(config, test_file=test_file,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
        )
    elif config.DATASET.DATASET=="300W":
        test_file="./data/300w/face_landmarks_300w_valid_common.csv"
        common_loader = DataLoader(
        dataset=dataset_type(config,test_file=test_file,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
        )
        test_file="./data/300w/face_landmarks_300w_valid_challenge.csv"

        challenge_loader = DataLoader(
        dataset=dataset_type(config,test_file=test_file,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
        )
        test_file="./data/300w/face_landmarks_300w_test.csv"

        test_loader = DataLoader(
        dataset=dataset_type(config,test_file=test_file,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
        )

    model.cpu()
    pruning.measure_sparsity(model)
    model.cuda()

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum").cuda(gpus[0])
    print("test")
    epoch=0
    if config.DATASET.DATASET=="WFLW":
        print("blur")
        nme, predictions = function.validate(config, blur_loader, model, criterion, epoch, writer_dict, loss_function)
        print("expression")
        nme, predictions = function.validate(config, expression_loader, model, criterion, epoch, writer_dict, loss_function)
        print("ill")
        nme, predictions = function.validate(config, ill_loader, model, criterion, epoch, writer_dict, loss_function)
        print("large")
        nme, predictions = function.validate(config, large_loader, model, criterion, epoch, writer_dict, loss_function)
        print("make")
        nme, predictions = function.validate(config, make_loader, model, criterion, epoch, writer_dict, loss_function)
        print("occ")
        nme, predictions = function.validate(config, occ_loader, model, criterion, epoch, writer_dict, loss_function)
    elif config.DATASET.DATASET=="AFLW":
        nme, predictions = function.validate(config, front_loader, model, criterion, epoch, writer_dict, loss_function)
    elif config.DATASET.DATASET=="300W":
        print("Test")
        nme, predictions = function.validate(config, test_loader, model, criterion, epoch, writer_dict, loss_function)
        print("common")
        nme, predictions = function.validate(config, common_loader, model, criterion, epoch, writer_dict, loss_function)
        print("challenge")
        nme, predictions = function.validate(config, challenge_loader, model, criterion, epoch, writer_dict, loss_function)
    nme, predictions = function.validate(config, val_loader, model, criterion, epoch, writer_dict, loss_function)


if __name__ == '__main__':
    main()










