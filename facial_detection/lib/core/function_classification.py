# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from .evaluation import decode_preds, compute_nme

logger = logging.getLogger(__name__)

def convert_new_old(encode,num_bits,di): # based on correlation
    encode = (encode.sign())
    t = torch.matmul(encode,di)
    t = torch.clamp(t,min=0.0)
    tsum=torch.reshape(torch.sum(t,dim=3),(t.size(0),t.size(1),t.size(2),1))
    t=t/tsum
    _,ts = torch.max(t,dim=3)
    return (ts,ts,ts)
def convert_new_tanh(encode,num_bits,di): # based on correlation
    encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    _,ts = torch.max(t,dim=3)
    return (ts,ts,ts)
def convert_new_tanh_loss(encode,num_bits,di): # based on correlation
    encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    return (t)
def convert_new(encode,num_bits,di): # based on correlation
    #encode = torch.tanh(encode)
    #t = torch.matmul(encode,di)
    _,ts = torch.max(encode,dim=3)
    return (ts,ts,ts)
def convert_new_loss(encode,num_bits,di): # based on correlation
    #encode = torch.tanh(encode)
    #t = torch.matmul(encode,di)
    return (encode)

def convert_soft_tanh(encode,num_bits,di,arr):
    encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    #t= t*3
    s = nn.Softmax(dim=3)
    t = s(t)
    t = t* arr
    ts = torch.sum(t,dim=3)
    return (ts)

def convert_soft(encode,num_bits,di,arr):
    #encode = torch.tanh(encode)
    #t = torch.matmul(encode,di)
    s = nn.Softmax(dim=3)
    t = s(encode)
    t = t* arr
    ts = torch.sum(t,dim=3)
    return (ts)

def convert_belj(encode,num_bits):
    arr=torch.tensor(range(num_bits,0,-1)).cuda()
    arrs = torch.tensor(range(1,num_bits+1,1)).cuda()
    encode = (encode.sign()+1)/2
    temp=encode*arr
    ts,_ = torch.max(temp,dim=3)
    temp=encode*arrs
    temp,_ = torch.max(temp,dim=3)
    te=128-temp
    return (ts+te)

def convert_belu(encode,num_bits):
    encode = (encode.sign()+1)/2
    number=torch.sum(encode,dim=3)
    number=number.cpu()
    return number
def convert_be1jmj(encode,num_bits):
    arr=torch.tensor(range(num_bits-1,-1,-1)).cuda()
    arrs = torch.tensor(range(1,num_bits+1,1)).cuda()
    arrs[num_bits-1]=0
    finmult = torch.zeros((num_bits)).cuda()
    finmult[num_bits-1]=2
    encode = (encode.sign()+1)/2
    temp=encode*arr
    ts,_ = torch.max(temp,dim=3)
    temp=encode*arrs
    temp,_ = torch.max(temp,dim=3)
    te=num_bits-1-temp
    temp=encode*finmult
    mult = torch.sum(temp,dim=3)-1
    mult1 = torch.sum(temp,dim=3)/2
    return (mult1*255-mult*ts-mult*te)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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
        self.avg = self.sum / self.count


def train(config, train_loader, model, criterion, optimizer,
          epoch, writer_dict, loss_fuction):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #print(config)
    model.train()
    nme_count = 0
    nme_batch_sum = 0
    critCE=torch.nn.CrossEntropyLoss(reduction="sum")
    critMSE=torch.nn.MSELoss(size_average=True)
    critL1=torch.nn.L1Loss(size_average=True)
    critMSEsum=torch.nn.MSELoss(reduction="sum")
    critL1sum=torch.nn.L1Loss(reduction="sum")
    end = time.time()
    arr=torch.tensor(range(5,256,10)).cuda((config.GPUS)[0])
    arrs = torch.tensor(range(1,66,1)).cuda((config.GPUS)[0])
    arrs[64]=0
    finmult = torch.zeros((65)).cuda()
    finmult[64]=2
    di=pickle.load(open(config.CODE.CODE_TENSOR,"rb"))
    di=torch.transpose(di,0,1).cuda()
    for i, (inp, target, meta) in tqdm(enumerate(train_loader)):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp.cuda((config.GPUS)[0]))
        ##target = target.cuda(non_blocking=True)
        target=torch.flatten((meta['codetpts'].cuda((config.GPUS)[0])), start_dim=1)
        if loss_fuction=="bce":
            loss = criterion(output, target)
        elif loss_fuction=="ce":
            outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
            targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
            t = (convert_new_loss(outputnew,int(config.CODE.CODE_BITS),di))
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda()
            loss= critCE(t.permute(0,3,1,2),torch.reshape(targets.long(),(t.size(0),t.size(1),t.size(2))).permute(0,1,2))
        elif loss_fuction=="ce_soft":
            outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
            targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
            t = (convert_new_loss(outputnew,int(config.CODE.CODE_BITS),di))
            targets = torch.floor(torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda()/10)
            loss= critCE(t.permute(0,3,1,2),torch.reshape(targets.long(),(t.size(0),t.size(1),t.size(2))).permute(0,1,2))
        elif loss_fuction=="ce_tanh":
            outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
            targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
            t = (convert_new_tanh_loss(outputnew,int(config.CODE.CODE_BITS),di))
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda()
            loss= critCE(t.permute(0,3,1,2),torch.reshape(targets.long(),(t.size(0),t.size(1),t.size(2))).permute(0,1,2))
        elif loss_fuction=="mse":
            outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
            targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
            t = convert_soft(outputnew,int(config.BITS/2),di,arr) 
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
            loss= critMSE(t,targets) 
        elif loss_fuction=="L1":
            outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
            targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
            t = convert_soft(outputnew,int(config.BITS/2),di,arr) 
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
            loss= critL1(t,targets) 
        elif loss_fuction=="msesum":
            outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
            targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
            t = convert_soft(outputnew,int(config.BITS/2),di,arr) 
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
            loss= critMSEsum(t,targets) 
        elif loss_fuction=="L1sum":
            outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
            targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
            t = convert_soft(outputnew,int(config.BITS/2),di,arr) 
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
            loss= critL1sum(t,targets) 
        elif loss_fuction=="mse_tanh":
            outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
            targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
            t = convert_soft_tanh(outputnew,int(config.BITS/2),di,arr) 
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
            loss= critMSE(t,targets) 
        #targets = convert(targetnew,int(config.BITS/2),arr,arrs)
        #t=torch.rand((loss.size(0),loss.size(1)),device=(config.GPUS)[0])
        #loss = (loss*t).sum()
        # NME
        ##score_map = output.data.cpu()
        ###output= output.data.cpu()
        if False:
            output = torch.reshape(output,(output.size(0),int(output.size(1)/config.BITS),2,int(config.BITS/2)))
            preds = convert(output,int(config.BITS/2),arr,arrs)
            ##preds=preds.data.cpu()
            ##preds=decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            preds=decode_preds(preds, meta['center'], meta['scale'], [config.BITS, config.BITS])

            nme_batch = compute_nme(preds.detach(), meta)
            nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
            nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    if False:
        nme = nme_batch_sum / nme_count
        msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'\
            .format(epoch, batch_time.avg, losses.avg, nme)
        logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict, loss_fuction):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    critCE=torch.nn.CrossEntropyLoss(reduction="sum")
    critMSE=torch.nn.MSELoss(size_average=True)
    critL1=torch.nn.L1Loss(size_average=True)
    critMSEsum=torch.nn.MSELoss(reduction="sum")
    critL1sum=torch.nn.L1Loss(reduction="sum")
    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    nme_batch_softtanh = 0
    nme_batch_cortanh = 0
    nme_batch_soft = 0
    nme_batch_cor = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()
    arr=torch.tensor(range(5,256,10)).cuda((config.GPUS)[0])
    di=pickle.load(open(config.CODE.CODE_TENSOR,"rb"))
    di=torch.transpose(di,0,1).cuda()
    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp.cuda((config.GPUS)[0]))
            ##target = target.cuda(non_blocking=True)
            #target=torch.flatten((meta['codetpts'].cuda((config.GPUS)[0])), start_dim=1)
            #loss = criterion(output, target)
            target=torch.flatten((meta['codetpts'].cuda((config.GPUS)[0])), start_dim=1)
            if loss_fuction=="bce":
                loss = criterion(output, target)
            elif loss_fuction=="ce":
                outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
                targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
                t = (convert_new_loss(outputnew,int(config.CODE.CODE_BITS),di))
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda()
                loss= critCE(t.permute(0,3,1,2),torch.reshape(targets.long(),(t.size(0),t.size(1),t.size(2))).permute(0,1,2))
            elif loss_fuction=="ce_soft":
                outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
                targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
                t = (convert_new_loss(outputnew,int(config.CODE.CODE_BITS),di))
                targets = torch.floor(torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda()/10)
                loss= critCE(t.permute(0,3,1,2),torch.reshape(targets.long(),(t.size(0),t.size(1),t.size(2))).permute(0,1,2))
            elif loss_fuction=="ce_tanh":
                outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
                targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
                t = (convert_new_tanh_loss(outputnew,int(config.CODE.CODE_BITS),di))
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda()
                loss= critCE(t.permute(0,3,1,2),torch.reshape(targets.long(),(t.size(0),t.size(1),t.size(2))).permute(0,1,2))
            elif loss_fuction=="mse":
                outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
                targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
                t = convert_soft(outputnew,int(config.BITS/2),di,arr) 
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
                loss= critMSE(t,targets) 
            elif loss_fuction=="L1":
                outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
                targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
                t = convert_soft(outputnew,int(config.BITS/2),di,arr) 
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
                loss= critL1(t,targets) 
            elif loss_fuction=="msesum":
                outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
                targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
                t = convert_soft(outputnew,int(config.BITS/2),di,arr) 
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
                loss= critMSEsum(t,targets) 
            elif loss_fuction=="L1sum":
                outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
                targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
                t = convert_soft(outputnew,int(config.BITS/2),di,arr) 
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
                loss= critL1sum(t,targets) 
            elif loss_fuction=="mse_tanh":
                outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
                targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS))) 
                t = convert_soft_tanh(outputnew,int(config.BITS/2),di,arr) 
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
                loss= critMSE(t,targets) 
                #t=torch.rand((loss.size(0),loss.size(1)),device=(config.GPUS)[0])
                #loss = (loss*t).sum()

            # NME
            ##score_map = output.data.cpu()
            ###output= output.data.cpu()
            output = torch.reshape(output,(output.size(0),int(output.size(1)/int(config.CODE.CODE_BITS*2)),2,config.CODE.CODE_BITS))
            _,_,preds = convert_new(output,int(config.BITS/2),di)
            preds=decode_preds(preds, meta['center'], meta['scale'], [config.BITS, config.BITS])
            nme_temp_cortanh = compute_nme(preds.detach(), meta)


            ##preds=preds.data.cpu()
            ##preds=decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            ## preds=decode_preds(preds, meta['center'], meta['scale'], [64, 64])
            #decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            # Failure Rate under different threshold
            failure_008 = (nme_temp_cortanh > 0.08).sum()
            failure_010 = (nme_temp_cortanh > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_cortanh += np.sum(nme_temp_cortanh)
            nme_count = nme_count + preds.size(0)
            for n in range(output.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    nme_cor_tanh = nme_batch_cortanh / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg_new_tanh = 'Test_cor_tanh Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme_cor_tanh,
                                failure_008_rate, failure_010_rate)
    logger.info(msg_new_tanh)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme_cor_tanh, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme_cor_tanh, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp.cuda((config.GPUS)[0]))
            ##target = target.cuda(non_blocking=True)
            target=torch.flatten((meta['codetpts'].cuda((config.GPUS)[0])), start_dim=1)
            loss = criterion(output, target)

            #t=torch.rand((loss.size(0),loss.size(1)),device=(config.GPUS)[0])
            #loss = (loss*t).sum()
            # NME
            ##score_map = output.data.cpu()
            output= output.data.cpu()
            output = torch.reshape(output,(output.size(0),int(output.size(1)/130),2,65))
            preds = convert(output,65,arr,arrs,finmult)
            ##preds=preds.data.cpu()
            ##preds=decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            preds=decode_preds(preds, meta['center'], meta['scale'], [config.BITS, config.BITS])
            ##score_map = output.data.cpu()
            ##preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            # NME
            print(preds)
            #print(target)
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions

from pathlib import Path
import os
import pickle
def log_ekn(config, val_loader, model,epoch,cfg_name,title,suf):
    root_output_dir = Path(config.OUTPUT_DIR)
    dataset = config.DATASET.DATASET
    cfg_name = (os.path.basename(cfg_name).split('.')[0])+str(config.TRAIN.LR)+str(config.SUF)+str(suf)

    final_output_dir = root_output_dir / dataset / cfg_name/"accuracy/"
    if not final_output_dir.exists():
        final_output_dir.mkdir() 
    final_output_dir=str(final_output_dir)
    f=open(final_output_dir+"/"+title+"_accuracy_log.csv", "a")
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    acc =torch.zeros((num_classes*2,config.BITS,int(config.CODE.CODE_BITS)))
    count =torch.zeros((num_classes*2,config.BITS,int(config.CODE.CODE_BITS)))

    with torch.no_grad():
        for i, (inp, target, meta) in tqdm(enumerate(val_loader)):
            output = model(inp.cuda((config.GPUS)[0]))
            ##target = target.cuda(non_blocking=True)
            target=torch.flatten((meta['codetpts'].cuda((config.GPUS)[0])), start_dim=1)
            output = torch.reshape(output,(output.size(0),int(output.size(1)/(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
            if True:
                 encode= (output.sign()+1)/2
                 correcttarget = torch.reshape(target,(target.size(0),int(target.size(1)/(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
                 diff  = (encode==correcttarget)
                 for v in range(0,diff.size(0)):
                     for label in range(0,num_classes*2):
                         l1=(label%num_classes)
                         l2=int(label/num_classes)
                         value=int((meta['tpts'])[v][l1][l2])
                         value=min(255,value)
                         value=max(0,value)
                         temp=diff[v:v+1,l1:l1+1,l2:l2+1,:]
                         temp=torch.flatten(temp)
                         acc[label][value]+=temp.cpu()
                         count[label][value]+=torch.ones((int(config.CODE.CODE_BITS)))
    
    if True:
        with open(final_output_dir+"/"+str(epoch)+"_"+title+".pkl", "wb") as fout:
            pickle.dump((acc,count), fout)
        print("Overall accuracy(%)")
        print(100* torch.sum(torch.flatten(acc))/torch.sum(torch.flatten(count)))
        print("label wise accuracy (%)")
        f.write("%s,%s,%s\n"%(str(epoch),str(100* torch.sum(torch.flatten(acc))/torch.sum(torch.flatten(count))),str(100* torch.sum(torch.sum(acc,dim=1),dim=1)/torch.sum(torch.sum(count,dim=1),dim=1))))
        print(100* torch.sum(torch.sum(acc,dim=1),dim=1)/torch.sum(torch.sum(count,dim=1),dim=1))
    plot_ekl(acc,count,int(config.CODE.CODE_BITS),epoch,title,final_output_dir)
    f.close()
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def plot_ekl(error,count, bits,epoch,title,final_output_dir):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    flatten_label=torch.sum(error,dim=0)
    flatten_count=torch.sum(count,dim=0)
    labels=list(range(0,256))
    prob=flatten_label/flatten_count

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 3}

    plt.rc('font', **font)
    xr = math.ceil(math.sqrt(bits))
    yr = math.ceil(bits/xr)
    for i in range(1,bits+1):
        #ax = fig.add_subplot(1, int(bits), i)
        ax = fig.add_subplot(xr,yr, i)
        ax.plot(labels, 1-prob[:,i-1:i], marker='o',linewidth=0, color="crimson", markersize=0.15, label="Empirical"+str(i))
        ax.axvline(x=(bits-i+1),c="green",linewidth=0.5,linestyle="--")
        ax.axvline(x=(bits*2-i+1),c="green",linewidth=0.5,linestyle="--")
        plt.text(100, 0.4 , "C"+str(i), rotation=0, verticalalignment='center',color="blue",fontsize="3")
        #if int(((i-1))+1)%16==1:
        ax.set_ylabel('Error probability',fontsize="3")
        if i==bits:
            ax.set_xlabel('Label',fontsize="3")
    plt.savefig(final_output_dir+"/"+str(epoch)+"_"+title+"_ekn_plots.pdf")


