from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import pickle
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
    t = torch.matmul(encode,di)
    _,ts = torch.max(t,dim=3)
    return (ts,ts,ts)
def convert_new_loss(encode,num_bits,di): # based on correlation
    #encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    return (t)

def convert_new_loss_soft(encode,num_bits,di): # based on correlation
    #encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    s = nn.Softmax(dim=3)
    t = s(t)
    return (t)

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
    t = torch.matmul(encode,di)
    s = nn.Softmax(dim=3)
    t = s(t)
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
    critMSE=torch.nn.MSELoss(reduction="mean")
    critL1=torch.nn.L1Loss(reduction="mean")
    critMSEsum=torch.nn.MSELoss(reduction="sum")
    critL1sum=torch.nn.L1Loss(reduction="sum")
    end = time.time()
    arr=torch.tensor(range(0,256,1)).cuda((config.GPUS)[0])
    arrs = torch.tensor(range(1,66,1)).cuda((config.GPUS)[0])
    arrs[64]=0
    finmult = torch.zeros((65)).cuda()
    finmult[64]=2
    di=pickle.load(open(config.CODE.CODE_TENSOR,"rb"))
    di=torch.transpose(di,0,1).cuda()
    for i, (inp, target, meta) in tqdm(enumerate(train_loader)):
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
    nme_batch_soft = 0
    nme_batch_cor = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()
    arr=torch.tensor(range(0,256,1)).cuda((config.GPUS)[0])
    di=pickle.load(open(config.CODE.CODE_TENSOR,"rb"))
    di=torch.transpose(di,0,1).cuda()
    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp.cuda((config.GPUS)[0]))
            target=torch.flatten((meta['codetpts'].cuda((config.GPUS)[0])), start_dim=1)
            if loss_fuction=="bce":
                loss = criterion(output, target)
            elif loss_fuction=="ce":
                outputnew = torch.reshape(output,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
                targetnew = torch.reshape(target,(output.size(0),int(output.size(1)/int(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
                t = (convert_new_loss(outputnew,int(config.CODE.CODE_BITS),di))
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

            # NME
            ##score_map = output.data.cpu()
            ###output= output.data.cpu()
            output = torch.reshape(output,(output.size(0),int(output.size(1)/int(config.CODE.CODE_BITS*2)),2,config.CODE.CODE_BITS))
            if config.CODE.CODE_NAME=="BELJ":
                 preds = convert_belj(output,int(config.CODE.CODE_BITS))
            elif config.CODE.CODE_NAME=="BELU":
                 preds = convert_belu(output,int(config.BITS/2))
            else:
                 _,_,preds = convert_new_tanh(output,int(config.BITS/2),di)
            preds=decode_preds(preds, meta['center'], meta['scale'], [config.BITS, config.BITS])
            nme_temp = compute_nme(preds.detach(), meta)

            preds = convert_soft(output,int(config.BITS/2),di,arr)
            preds=decode_preds(preds, meta['center'], meta['scale'], [config.BITS, config.BITS])
            nme_temp_soft = compute_nme(preds.detach(), meta)

            _,_,preds = convert_new(output,int(config.BITS/2),di)
            preds=decode_preds(preds, meta['center'], meta['scale'], [config.BITS, config.BITS])
            nme_temp_cor = compute_nme(preds.detach(), meta)

            # NME
            # Failure Rate under different threshold
            failure_008 = (nme_temp_soft > 0.08).sum()
            failure_010 = (nme_temp_soft > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_batch_soft += np.sum(nme_temp_soft)
            nme_batch_cor += np.sum(nme_temp_cor)
            nme_count = nme_count + preds.size(0)
            for n in range(output.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    nme_soft = nme_batch_soft / nme_count
    nme_cor  = nme_batch_cor / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test_sf Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    msg_new1 = 'Test_soft Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme_soft,
                                failure_008_rate, failure_010_rate)
    logger.info(msg_new1)

    msg_new_tanh1 = 'Test_cor Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme_cor,
                                failure_008_rate, failure_010_rate)
    logger.info(msg_new_tanh1)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions

