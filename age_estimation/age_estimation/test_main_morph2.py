import argparse
import os
import pprint
import shutil

import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from ageresnet import MORPH2Dataset, get_age_resnet, AgeL1Loss, AgeBCELoss, train, validate, test, RegressionTransform, appendToCsv, l1_prune_network, Logger

BATCH_SIZE = 64
NUM_WORKERS=4

# cudnn related setting
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

BASE_DIR="../dataset/morph2/"

TRAIN_CSV_PATH = BASE_DIR + 'morph2_train.csv'
VALID_CSV_PATH = BASE_DIR + 'morph2_valid.csv'
TEST_CSV_PATH = BASE_DIR + 'morph2_test.csv'
IMAGE_PATH = BASE_DIR + '/morph2-aligned'

def test_main(args, output_transform, result_transform, criterion, num_classes=1):
    normalize = transforms.Normalize((0.5958, 0.4637, 0.4065), (0.2693, 0.2409, 0.2352))

    custom_transform2 = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.CenterCrop((224, 224)),
                                        transforms.ToTensor(),
                                        normalize])

    test_dataset = MORPH2Dataset(csv_path=TEST_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform2,
                                output_transform=output_transform)

    valid_dataset = MORPH2Dataset(csv_path=VALID_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform2,
                                output_transform=output_transform)

    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS)
    
    torch.manual_seed(0)
    np.random.seed(0)

    train_loader=None

    test_loop(args, num_classes, result_transform, train_loader, valid_loader, test_loader, criterion)

def test_loop(args, num_classes, result_transform, train_loader, valid_loader, test_loader, criterion):
    baseLabel = args.dataset + args.transform

    if args.label is None:
        label = baseLabel
    else:
        label = args.label

    sys.stdout = Logger(label)

    gpus = [int(i) for i in args.gpus.split(',')]
    model = get_age_resnet(num_classes, 50, True, 128)
    # model.load_state_dict(torch.load('iwtest.pth.tar'))

    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model.load_state_dict(torch.load(args.ckpt))



    # state_dict = model.state_dict()
    # pretrained_state_dict = torch.load('iwcnt.pth.tar')
    # pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in state_dict and 'final_layer' not in k}
    # state_dict.update(pretrained_state_dict)
    # model.load_state_dict(state_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [30, 40], 0.1
        )


    min_train_mae = 999
    min_valid_mae = 999

    
    
    perf, loss, cs5 = test(args, test_loader, model, criterion, result_transform)

    msg =   'Test: \t'\
            'TestLoss: {0:.5f}\t'\
            'TestMae: {1:.5f}\t'\
            'TestCS5: {2:.5f}\t'.format(
                loss, perf, cs5
            )
    print(msg)

def finetune_loop(args, num_classes, result_transform, train_loader, valid_loader, test_loader, criterion):
    baseLabel = args.dataset + args.transform

    if args.label is None:
        label = baseLabel + str(args.sparsity)
    else:
        label = args.label

    sys.stdout = Logger(label)
    
    gpus = [int(i) for i in args.gpus.split(',')]
    model = get_age_resnet(num_classes, 18, True, 64)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model.load_state_dict(torch.load(baseLabel + '.pth.tar'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [90, 120], 0.1
        )

    min_train_mae = 999
    min_valid_mae = 999

    l1_prune_network(model, amount=args.sparsity)

    for epoch in range(args.num_epochs):
        lr_scheduler.step()

        tperf, tloss, tcs5 = train(args, train_loader, model, criterion, optimizer, epoch, result_transform)

        perf, loss, cs5 = validate(args, valid_loader, model, criterion, epoch, result_transform)

        if tperf < min_train_mae:
            min_train_mae = tperf
        if perf < min_valid_mae:
            min_valid_mae = perf

        msg =   'Epoch: [{0}]\t'\
                'TrainLoss: {1:.5f}\t'\
                'TrainMae: {2:.5f}\t'\
                'TrainCS5: {3:.5f}\t'\
                'ValidLoss: {4:.5f}\t'\
                'ValidMAE: {5:.5f}\t'\
                'ValidCS5: {6:.5f}\t'.format(
                    epoch, tloss, tperf, tcs5, loss, perf, cs5
                )
        print(msg)
    
    perf, loss, cs5 = test(args, test_loader, model, criterion, result_transform)

    appendToCsv(baseLabel + '.csv', ['sparsity', 'train', 'valid', 'test'], [args.sparsity, min_train_mae, min_valid_mae, perf])

    msg =   'Test: \t'\
            'TestLoss: {0:.5f}\t'\
            'TestMae: {1:.5f}\t'\
            'TestCS5: {2:.5f}\t'.format(
                loss, perf, cs5
            )
    print(msg)
    
    torch.save(model.state_dict(), label + '.pth.tar')
