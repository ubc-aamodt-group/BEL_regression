import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models_new as hopenet
import helper_bcd as helper
import datasets
import torch.utils.model_zoo as model_zoo
from torchsummary import summary
from tqdm import tqdm
import pickle
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--prune_amount', dest='prune_amount', help='prune_amount.',
          default=0.0, type=float)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--test_filename_list', dest='test_filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--val_filename_list', dest='val_filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--arch', dest='arch', help='architecture_type: resnet or resnet_stage.', default='resnet', type=str)
    parser.add_argument('--num_bits', dest='num_bits', help='num_bits.',
          default=6, type=int)
    parser.add_argument('--code_bits', dest='code_bits', help='code_bits. biwi:150 for U, other is scaled accordiignly',
          default=100, type=int)
    parser.add_argument('--code', dest='code', help='code_type: u or j or b1jdj or b2jdj or hex16 or had', default='u', type=str)
    parser.add_argument('--loss', dest='loss_func', help='loss function type: bce or ce or mse or l1', default='bce', type=str)
    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_angles]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    prune_amount=args.prune_amount
    loss_func=args.loss_func
    if (os.environ['TMPDIR']==""):
        os.environ['TMPDIR']="."

    if not os.path.exists(os.environ['TMPDIR']+'/output/snapshots'):
        os.makedirs(os.environ['TMPDIR']+'/output/snapshots')
    di=pickle.load(open("code_pkl/bel"+args.code+"_200_tensor.pkl","rb"))
    dis=pickle.load(open("code_pkl/bel"+args.code+"_200_tensor.pkl","rb"))
    # ResNet50

    model = hopenet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3,num_bits=args.num_bits, code=args.code)
    load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_dir=".cache/"))

    print("Loading data")
    

    transformations = transforms.Compose([transforms.Scale(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
        val_pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.val_filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds_bcd(args.data_dir, "datasets/300W_LP/overall_filelist_clean.txt", transformations,code_bits=args.code_bits, code=args.code)
        test_pose_dataset = datasets.AFLW2000_bcd("datasets/AFLW2000/", "datasets/AFLW2000/overall_filelist_clean.txt", transformations,code_bits=args.code_bits, code=args.code)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, transformations)
        val_pose_dataset = datasets.AFLW2000(args.data_dir, args.val_filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI_bcd(args.data_dir, args.filename_list, transformations,num_bits=args.num_bits)
        test_pose_dataset = datasets.BIWI_bcd(args.data_dir, args.test_filename_list, transformations,num_bits=args.num_bits)
        #val_pose_dataset = datasets.BIWI(args.data_dir, args.val_filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations)
        val_pose_dataset = datasets.AFLW(args.data_dir, args.val_filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, transformations)
        val_pose_dataset = datasets.AFLW_aug(args.data_dir, args.val_filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations)
        val_pose_dataset = datasets.AFW(args.data_dir, args.val_filename_list, transformations)
    else:
        print("Error: not a valid dataset name")
        sys.exit()
    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)



    helper.train(model,train_loader, test_loader, pose_dataset, args.output_string, gpu, args.arch,args.lr,20,batch_size,195,args.num_bits,args.code_bits,args.code,loss_func,di,dis)
