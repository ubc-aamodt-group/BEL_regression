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
from lib.core import function
from lib.utils import utils
from torchsummary import summary
from torchvision import transforms
import torchvision
import torch.utils.model_zoo as model_zoo
import pruning
import os
import shutil
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)
def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--code', help='code configuration filename',
                        required=True, type=str)
    parser.add_argument('--suf', help='suffix for logfile',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    update_config_code(config, args)
    return args


def main():

    args = parse_args()
    suf = args.suf+config.CODE.CODE_NAME+"_6L1"

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train',suf)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    model = models.get_face_alignment_net6(config)
    #eval('cls_models.'+'cls_hrnet'+'.get_cls_net')(config)
    gpus = list(config.GPUS)
    #model = models.ResNetN(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3],"cuda:"+str(gpus[0]),num_classes=2*(config.MODEL.NUM_JOINTS),num_bits=config.MODEL.LAYER)
    #saved_state_dict = torch.load("output/snapshots/resnet_bcd_8.pkl")
    #load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_dir="../.cache/"))
    

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print(gpus)
    model = model.cuda(gpus[0])
    #print(summary(model,(3,256,256)))
    # loss
    criterion = torch.nn.L1Loss(size_average=True).cuda(gpus[0])
    #criterion = torch.nn.SmoothL1Loss(size_average=True).cuda(gpus[0])

    optimizer = utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    dataset_type = get_dataset(config)

    train_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)
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

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        print(optimizer.param_groups[0]['lr'])
        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict)
        #function.log_ekn(config, train_loader, model,epoch,args.cfg,"train")
        #function.log_ekn(config, val_loader, model,epoch,args.cfg,"valid")
        # evaluate
        if (epoch+1)%10==0 or epoch>55:
            #function.log_ekn(config, train_loader, model,epoch,args.cfg,"train")
            #function.log_ekn(config, val_loader, model,epoch,args.cfg,"valid")
            print("test")
            if config.DATASET.DATASET=="WFLW":
                print("blur")
                nme, predictions = function.validate(config, blur_loader, model, criterion, epoch, writer_dict)
                print("expression")
                nme, predictions = function.validate(config, expression_loader, model, criterion, epoch, writer_dict)
                print("ill")
                nme, predictions = function.validate(config, ill_loader, model, criterion, epoch, writer_dict)
                print("large")
                nme, predictions = function.validate(config, large_loader, model, criterion, epoch, writer_dict)
                print("make")
                nme, predictions = function.validate(config, make_loader, model, criterion, epoch, writer_dict)
                print("occ")
                nme, predictions = function.validate(config, occ_loader, model, criterion, epoch, writer_dict)
            elif config.DATASET.DATASET=="AFLW":
                nme, predictions = function.validate(config, front_loader, model, criterion, epoch, writer_dict)
            elif config.DATASET.DATASET=="300W":
                print("Test")
                nme, predictions = function.validate(config, test_loader, model, criterion, epoch, writer_dict)
                print("common")
                nme, predictions = function.validate(config, common_loader, model, criterion, epoch, writer_dict)
                print("challenge")
                nme, predictions = function.validate(config, challenge_loader, model, criterion, epoch, writer_dict)
            nme, predictions = function.validate(config, val_loader, model, criterion, epoch, writer_dict)

            is_best = nme < best_nme
            best_nme = min(nme, best_nme)
    
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            print("best:", is_best)
            torch.save(model.state_dict(), final_output_dir+"/checkpoint"+str(epoch%2)+".pth")
            if is_best:
                 torch.save(model.state_dict(), final_output_dir+"/model_best.pth")
            #utils.save_checkpoint(
            #    {"state_dict": model,
            #     "epoch": epoch + 1,
            #     "best_nme": best_nme,
            #     "optimizer": optimizer.state_dict(),
            #     }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch%3))
    nme, predictions = function.validate(config, train_loader, model, criterion, epoch, writer_dict)
    model.cpu()
    pruning.measure_sparsity(model)
    sys.exit()


if __name__ == '__main__':
    main()










