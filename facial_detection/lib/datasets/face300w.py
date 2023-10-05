# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random
import pickle
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

def temp_code(num,num_bits):
    num=max(0,num)
    bits= int(num_bits/2)
    a= torch.zeros([bits],dtype=torch.long)
    for i in range(0,bits):
        if num > (bits-i-1) and num <= num_bits-i-1:
            a[i] =1
    return a

class Face300W(data.Dataset):

    def __init__(self, cfg, test_file="",is_train=True, transform=None,drop=0.0):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
            if drop>0:
                 r=str(drop).replace(".","o")+"_face"
                 self.csv_file = cfg.DATASET.TRAINSET.replace("face",r)
                 print(self.csv_file)
        else:
            self.csv_file = test_file#cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        self.bits=cfg.BITS
        self.code_bits=cfg.CODE.CODE_BITS
        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)
        self.nby2=pickle.load(open(cfg.CODE.CODE_PKL,"rb"))

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]

        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = pts.astype('float').reshape(-1, 2)

        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        #print(pts)
        codetpts=torch.zeros((nparts,2,self.code_bits))
        for i in range(nparts):
            i0=int(round(tpts[i][0].item()))
            i1=int(round(tpts[i][1].item()))
            i0=min(255,i0)
            i0=max(0,i0)
            i1=min(255,i1)
            i1=max(0,i1)
            codetpts[i][0]=self.nby2[i0]
            codetpts[i][1]=self.nby2[i1]
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts,'codetpts': codetpts}
        #print(tpts) ## tpts gives output in 0-64 range
        return img, target, meta


if __name__ == '__main__':

    pass
