import argparse
import os
import pprint
import shutil

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

MAELoss = nn.L1Loss()

def CS5calc(a, b):
    f = np.abs(b.detach().numpy()-a.detach().numpy())
    f = f <= 5
    return np.sum(f) / len(f)

def train(args, train_loader, model, criterion, optimizer, epoch, transform):
    losses = AverageMeter()
    maes = AverageMeter()
    cs5s = AverageMeter()

    model.train()

    for idx, (features, targets, levels) in enumerate(train_loader):
        targets = np.reshape(targets, (-1, 1))

        features = features.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).flatten()
        levels = levels.cuda(non_blocking=True)

        out = model(features)
        
        if args.loss == "mae" or args.loss == "mse":
            outp = transform.convert_continuous(out)
            loss = criterion(outp, targets.clone().detach().float())
        elif args.transform == "mc":
            loss = criterion(out, targets)
        elif args.loss == "ce":
            outp = transform.convert_discrete(out)
            loss = criterion(outp, targets)
        
        else:
            loss = criterion(out, levels)

        if epoch % 10 == 0:
            oute = transform(out)
            oute = oute.cpu()
            targets = targets.cpu()
            mae = MAELoss(oute.flatten(), targets)
            cs5 = CS5calc(oute.flatten(), targets)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        losses.update(loss.item(), features.size(0))
        if epoch % 10 == 0:
            maes.update(mae.item(), features.size(0))
            cs5s.update(cs5.item(), features.size(0))
        else:
            maes.update(0, features.size(0))
            cs5s.update(0, features.size(0))

        if idx % 100 == 0:
            msg =   'Epoch: [{0}][{1}/{2}]\t'\
                        'Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                        'MAE: {mae.val:.5f} ({mae.avg:.5f})\t'\
                        'CS5: {cs5.val:.5f} ({cs5.avg:.5f})'.format(
                            epoch, idx, len(train_loader), loss=losses, mae=maes, cs5=cs5s
                        )
            print(msg)
    return maes.avg, losses.avg, cs5s.avg
    

def validate(args, valid_loader, model, criterion, epoch, transform):
    losses = AverageMeter()
    maes = AverageMeter()
    cs5s = AverageMeter()

    model.eval()
    with torch.no_grad():
        for idx, (features, targets, levels) in enumerate(valid_loader):
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True).flatten()
            levels = levels.cuda(non_blocking=True)

            out = model(features)
            if args.loss == "mae" or args.loss == "mse":
                outp = transform.convert_continuous(out)
                loss = criterion(outp, targets)
            elif args.transform == "mc":
                loss = criterion(out, targets)
            elif args.loss == "ce":
                outp = transform.convert_discrete(out)
                loss = criterion(outp, targets)
            
            else:
                loss = criterion(out, levels)

            oute = transform(out)
            oute = oute.cpu()
            targets = targets.cpu()
            mae = MAELoss(oute.flatten(), targets)
            cs5 = CS5calc(oute.flatten(), targets)

            losses.update(loss.item(), features.size(0))
            maes.update(mae.item(), features.size(0))
            cs5s.update(cs5.item(), features.size(0))

            if idx % 100 == 0:
                msg =   'Epoch: [{0}][{1}/{2}]\t'\
                        'Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                        'MAE: {mae.val:.5f} ({mae.avg:.5f})\t'\
                        'CS5: {cs5.val:.5f} ({cs5.avg:.5f})'.format(
                            epoch, idx, len(valid_loader), loss=losses, mae=maes, cs5=cs5s
                        )
                print(msg)
    
    return maes.avg, losses.avg, cs5s.avg


def test(args, test_loader, model, criterion, transform):
    losses = AverageMeter()
    maes = AverageMeter()
    cs5s = AverageMeter()


    model.eval()
    with torch.no_grad():
        for idx, (features, targets, levels) in enumerate(test_loader):
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True).flatten()
            levels = levels.cuda(non_blocking=True)

            out = model(features)
            if args.loss == "mae" or args.loss == "mse":
                outp = transform.convert_continuous(out)
                loss = criterion(outp, targets)
            elif args.transform == "mc":
                loss = criterion(out, targets)
            elif args.loss == "ce":
                outp = transform.convert_discrete(out)
                loss = criterion(outp, targets)
            
            else:
                loss = criterion(out, levels)

            oute = transform(out)
            oute = oute.cpu()
            targets = targets.cpu()
            mae = MAELoss(oute.flatten(), targets)
            cs5 = CS5calc(oute.flatten(), targets)
            
            losses.update(loss.item(), features.size(0))
            maes.update(mae.item(), features.size(0))
            cs5s.update(cs5.item(), features.size(0))
            if idx % 100 == 0:
                msg =   'Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                        'MAE: {mae.val:.5f} ({mae.avg:.5f})\t'\
                        'CS5: {cs5.val:.5f} ({cs5.avg:.5f})'.format(
                            loss=losses, mae=maes, cs5=cs5s
                        )
                print(msg)
            # handle mae logic

    return maes.avg, losses.avg, cs5s.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
