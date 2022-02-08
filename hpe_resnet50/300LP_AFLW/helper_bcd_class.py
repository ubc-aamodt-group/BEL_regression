import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

import datasets, utils
from tqdm import tqdm
import pickle

mult=1

def convert_new_tanh(encode,num_bits,di): # based on correlation
    encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    _,ts = torch.max(t,dim=1)
    return ts
def convert_new_tanh_loss(encode,num_bits,di): # based on correlation
    encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    return (t)

def convert_cor(encode,num_bits,di): # based on correlation
    #t = torch.matmul(encode,di)
    _,ts = torch.max(encode,dim=1)
    return ts
def convert_cor_loss(encode,num_bits,di): # based on correlation
    #t = torch.matmul(encode,di)
    return (encode)

def convert_soft(encode,num_bits,di):
    arr = torch.tensor(range(0,num_bits,1)).cuda()
    #encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    s = nn.Softmax(dim=1)
    t = s(t)
    t = t* arr
    ts = torch.sum(t,dim=1)
    return (ts)

def convert_belj(encode,num_bits,di):
    num_bits=int(num_bits/2)
    arr=torch.tensor(range(num_bits,0,-1)).cuda()
    arrs = torch.tensor(range(1,num_bits+1,1)).cuda()
    encode = (encode.sign()+1)/2
    temp=encode*arr
    ts,_ = torch.max(temp,dim=1)
    temp=encode*arrs
    temp,_ = torch.max(temp,dim=1)
    te=num_bits-temp
    return (ts+te)

def convert_belu(encode,num_bits,di):
    encode = (encode.sign()+1)/2
    number=torch.sum(encode,dim=1)
    number=number.cpu()
    return number


def measure_MAE(cont_labels,yaw,pitch,roll,gpu,func,di,dis):
    #cont_labels=cont_labels.cuda(gpu)
    #print(yaw)
    
    #yaw = (yaw.sign()+1)/2
    #pitch = (pitch.sign()+1)/2
    #roll = (roll.sign()+1)/2

    yaw_predicted=func(yaw,200,di).cpu()
    pitch_predicted= func(pitch,200,di).cpu()
    roll_predicted=func(roll,200,dis).cpu()

    label_yaw = cont_labels[:,0].float()
    label_pitch = cont_labels[:,1].float()
    label_roll = cont_labels[:,2].float()
    yaw_error = torch.sum(torch.abs(yaw_predicted - label_yaw))
    pitch_error = torch.sum(torch.abs(pitch_predicted - label_pitch))
    roll_error = torch.sum(torch.abs(roll_predicted - label_roll))
    return yaw_error, pitch_error,roll_error
def measure(model,i,images,labels, cont_labels, name, yaw_error, pitch_error,roll_error,gpu,total):
    images = Variable(images).cuda(gpu)
    total += cont_labels.size(0)
    yaw,pitch,roll = (model(images))
    yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu)

    # Mean absolute error
    yaw_error += yerror
    pitch_error += perror
    roll_error += rerror
    return yaw_error, pitch_error,roll_error, total


def get_ignored_params(model,arch):
    # Generator function that yields ignored params.
    if arch=="vgg16":
        b=[]
    else:
        b = [model.conv1, model.bn1]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model,arch):
    # Generator function that yields params that will be optimized.
    if arch=="vgg16":
        b = [model.features]
    else:
        b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model,arch):
    # Generator function that yields fc layer params.
    if arch=="vgg16":
        b = [model.classifier, model.fc_angles]
    elif arch=="resnet_stage":
        b = [model.fc1y,model.fc1p,model.fc1r,
            model.fc2y,model.fc2p,model.fc2r,
            model.fc3y,model.fc3p,model.fc3r,
            model.fc4y,model.fc4p,model.fc4r,
            model.fc5y,model.fc5p,model.fc5r,
            model.fc6y,model.fc6p,model.fc6r,
            model.fc7y,model.fc7p,model.fc7r,
            model.fc8y,model.fc8p,model.fc8r,]
    else:
        b = [model.fc_angles_yaw,model.fc_angles_pitch,model.fc_angles_roll,model.yawm,model.pitchm,model.rollm]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def print_statement(yaw,pitch,roll,total,f,label):
    print(label+ ' error in degrees of the model on the ' + str(total) +
        ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f' % (yaw / total,
        pitch / total, roll / total))
    print("%s MAE= %.4f"% (label,((yaw / total)+(pitch / total)+(roll / total))/3))
    f.write(label+ ' error in degrees of the model on the ' + str(total) +
        ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f \n' % (yaw / total,
        pitch / total, roll / total))
    f.write("%s MAE= %.4f \n"% (label,((yaw / total)+(pitch / total)+(roll / total))/3))


def train(model,train_loader,test_loader, pose_dataset, output_string, gpu, arch,lr,num_epochs,batch_size,val_bound,num_bits,code_bits,code,loss_func,di,dis):
    f=open(os.environ['TMPDIR']+"/output/snapshots/" + output_string,"a")
    model.cuda(gpu)
    criterion = nn.BCEWithLogitsLoss(reduction="sum").cuda(gpu)
    optimizer = torch.optim.Adam([{'params': get_ignored_params(model,arch), 'lr': 0},
                                  {'params': get_non_ignored_params(model,arch), 'lr': lr},
                                  {'params': get_fc_params(model,arch), 'lr': lr * 5} ],
                                   lr = lr)
    critCE=torch.nn.CrossEntropyLoss(reduction="sum").cuda()
    critMSE=torch.nn.MSELoss(reduction="mean").cuda()
    critL1=torch.nn.L1Loss(reduction="mean").cuda()
    print('Ready to train network.')
    best_MAE=100
    convert_func={'u':convert_belu,'j':convert_belj,"b1jdj":convert_cor,"b2jdj":convert_cor,"hex16":convert_cor,"had":convert_cor}
    convert_sf=convert_func[code]
    f.write('Ready to train network.')
    #di=pickle.load(open("../belj_200_tensor.pkl","rb"))
    #dis=pickle.load(open("../belj_200_tensor.pkl","rb"))
    di=torch.transpose(di,0,1).cuda()
    dis=torch.transpose(dis,0,1).cuda()
    for epoch in range(num_epochs):
        model.train()
        if epoch==10:
            lr =lr/10
            optimizer = torch.optim.Adam([{'params': get_ignored_params(model,arch), 'lr': 0},
                                  {'params': get_non_ignored_params(model,arch), 'lr': lr},
                                  {'params': get_fc_params(model,arch), 'lr': lr * 5}],
                                   lr = lr)
        val_loss, train_loss, test_loss=0.0, 0.0, 0.0
        ttotal,tyaw_error ,tpitch_error ,troll_error = 0,.0,.0,.0
        tstotal,tsyaw_error ,tspitch_error ,tsroll_error = 0,.0,.0,.0
        ts1total,ts1yaw_error ,ts1pitch_error ,ts1roll_error = 0,.0,.0,.0
        ts2total,ts2yaw_error ,ts2pitch_error ,ts2roll_error = 0,.0,.0,.0
        ts3total,ts3yaw_error ,ts3pitch_error ,ts3roll_error = 0,.0,.0,.0


        for i, (images, labels, cont_labels, name, tyaw,tpitch,troll,tiyaw,tipitch,tiroll) in tqdm((enumerate(train_loader))):
            tyaw= tyaw.cuda(gpu)
            tpitch= tpitch.cuda(gpu)
            troll= troll.cuda(gpu)
            images = Variable(images).cuda(gpu)
            yaw,pitch,roll,rangles = (model(images))
            angles = torch.cat((yaw,pitch,roll),dim=1)
            bout = torch.cat((tyaw,tpitch,troll),dim=1)
            if loss_func=="bce":
                loss = criterion(angles.float(), bout.float())
            elif loss_func=="ce":
                ty = (convert_cor_loss(yaw,200,di))
                tp = (convert_cor_loss(pitch,200,di))
                tr = (convert_cor_loss(roll,200,dis))
                loss= 1.0*( critCE(ty,tiyaw.cuda()) + critCE(tp,tipitch.cuda()) + critCE(tr,tiroll.cuda()))
            elif loss_func=="mse":
                ty = (convert_soft(yaw,200,di))
                tp = (convert_soft(pitch,200,di))
                tr = (convert_soft(roll,200,dis))
                loss= 1.0*( critMSE(ty, cont_labels[:,0].float().cuda()) + critMSE(tp, cont_labels[:,1].float().cuda()) + critMSE(tr, cont_labels[:,2].float().cuda()))
            elif loss_func=="L1":
                ty = (convert_soft(yaw,200,di))
                tp = (convert_soft(pitch,200,di))
                tr = (convert_soft(roll,200,dis))
                loss= 1.0*( critL1(ty, cont_labels[:,0].float().cuda()) + critMSE(tp, cont_labels[:,1].float().cuda()) + critMSE(tr, cont_labels[:,2].float().cuda()))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=(loss.data).cpu().numpy()
   
            ttotal += cont_labels.size(0)
            yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_cor,di,dis)
            tyaw_error += yerror; tpitch_error += perror;   troll_error += rerror
        for j, (images, labels, cont_labels, name, tyaw,tpitch,troll) in tqdm((enumerate(test_loader))):
                model.eval()
                tyaw= tyaw.cuda(gpu)
                tpitch= tpitch.cuda(gpu)
                troll= troll.cuda(gpu)
                with torch.no_grad():
                    images = Variable(images).cuda(gpu)
                    yaw,pitch,roll,_ = (model(images))
                    angles = torch.cat((yaw,pitch,roll),dim=1)
                    bout = torch.cat((tyaw,tpitch,troll),dim=1)
                    loss = criterion(angles.float(), bout.float())
                    test_loss+=(loss.data).cpu().numpy()
                    tstotal += cont_labels.size(0)
                    #yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_sf,di,dis)
                    #tsyaw_error += yerror;       tspitch_error += perror;        tsroll_error += rerror
                    yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_cor,di,dis)
                    ts1yaw_error += yerror;       ts1pitch_error += perror;        ts1roll_error += rerror
                    #yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_soft,di,dis)
                    #ts2yaw_error += yerror;       ts2pitch_error += perror;        ts2roll_error += rerror
        cur_MAE= ((ts2yaw_error+ts2pitch_error+ts2roll_error)/tstotal)
        if cur_MAE<=best_MAE:
                 print("best:")
                 best_MAE=cur_MAE
                 torch.save(model.state_dict(),os.environ['TMPDIR']+"/output/snapshots/" + output_string+"_best_model.pth")
        torch.save(model.state_dict(),os.environ['TMPDIR']+"/output/snapshots/" + output_string+str(epoch%2)+"_model.pth")
        print_statement(tyaw_error,tpitch_error,troll_error,ttotal,f,"train ")

        #print_statement(tsyaw_error,tspitch_error,tsroll_error,tstotal,f,"test_sf ")
        print_statement(ts1yaw_error,ts1pitch_error,ts1roll_error,tstotal,f,"test_cor ")
        #print_statement(ts2yaw_error,ts2pitch_error,ts2roll_error,tstotal,f,"test_soft ")


    f.close()
def test(model,test_loader, pose_dataset, output_string, gpu, arch,lr,num_epochs,batch_size,val_bound,num_bits,code_bits,code,loss_func,di,dis):
    f=open(os.environ['TMPDIR']+"/"+output_string+"_test","w")
    model.cuda(gpu)
    di=torch.transpose(di,0,1).cuda()
    dis=torch.transpose(dis,0,1).cuda()
    ttotal,tyaw_error ,tpitch_error ,troll_error = 0,.0,.0,.0
    tstotal,tsyaw_error ,tspitch_error ,tsroll_error = 0,.0,.0,.0
    ts1total,ts1yaw_error ,ts1pitch_error ,ts1roll_error = 0,.0,.0,.0
    ts2total,ts2yaw_error ,ts2pitch_error ,ts2roll_error = 0,.0,.0,.0
    ts3total,ts3yaw_error ,ts3pitch_error ,ts3roll_error = 0,.0,.0,.0
    for j, (images, labels, cont_labels, name, tyaw,tpitch,troll) in tqdm((enumerate(test_loader))):
            model.eval()
            tyaw= tyaw.cuda(gpu)
            tpitch= tpitch.cuda(gpu)
            troll= troll.cuda(gpu)
            with torch.no_grad():
                images = Variable(images).cuda(gpu)
                yaw,pitch,roll,rangles = (model(images))
                angles = torch.cat((yaw,pitch,roll),dim=1)
                bout = torch.cat((tyaw,tpitch,troll),dim=1)
                loss = criterion(angles.float(), bout.float())
                test_loss+=(loss.data).cpu().numpy()
                tstotal += cont_labels.size(0)
                #yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_sf,di,dis)
                #tsyaw_error += yerror;       tspitch_error += perror;        tsroll_error += rerror
                yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_cor,di,dis)
                ts1yaw_error += yerror;       ts1pitch_error += perror;        ts1roll_error += rerror
                #yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_soft,di,dis)
                #ts2yaw_error += yerror;       ts2pitch_error += perror;        ts2roll_error += rerror


    #print_statement(tsyaw_error,tspitch_error,tsroll_error,tstotal,f,"test_sf ")
    print_statement(ts1yaw_error,ts1pitch_error,ts1roll_error,tstotal,f,"test_cor ")
    f.close()
def finetune(model,train_loader,test_loader,pose_dataset, output_string,gpu,arch,batch_size,val_bound,num_bits):
    lr=0.0001
    model.cuda(gpu)
    num_epochs=10
    lr=0.00001
    train(model,train_loader,test_loader,pose_dataset, output_string,gpu,arch,lr,num_epochs,batch_size,val_bound,num_bits)
    num_epochs=10
    train(model,train_loader,test_loader,pose_dataset, output_string,gpu,arch,lr,num_epochs,batch_size,val_bound,num_bits)
