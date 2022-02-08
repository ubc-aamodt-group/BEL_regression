import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image

import json_tricks as json
import numpy as np

class RegressionTransform(nn.Module):
    def __init__(self):
        super(RegressionTransform, self).__init__()
        
    def forward(self, x):
        return x

class MultiClassReverseTransform(nn.Module):
    def __init__(self):
        super(MultiClassReverseTransform, self).__init__()

    def forward(self, x):
        return torch.argmax(x, dim=1).float()

class TempTransform(nn.Module):
    def __init__(self, vrange):
        super(TempTransform, self).__init__()
        self.vrange = vrange

        
    def forward(self, x):
        a = torch.ones([self.vrange], dtype=torch.float32)
        for i in range(x):
            a[i] = 0
        return a

class TempReverseTransform(nn.Module):
    def __init__(self, vrange):
        super(TempReverseTransform, self).__init__()
        self.vrange = vrange
        
    def forward(self, x):
        x = (x.sign() + 1)/2

        N, D = x.shape
        num = torch.sum(x, dim=1)
        return num

class BCDTransform(nn.Module):
    def __init__(self, bits):
        super(BCDTransform, self).__init__()
        self.bits = bits
        
    def forward(self, x):
        binary = [0]*self.bits
        for i in range(self.bits):
            binary[i] = x % 2
            x //= 2
        return torch.tensor(binary, dtype=torch.float32)

class BCDReverseTransform(nn.Module):
    def __init__(self, bits):
        super(BCDReverseTransform, self).__init__()
        self.bits = bits
        
    def forward(self, x):
        x = (x.sign() + 1)/2
        N, D = x.shape
        binary = [0] * N

        for i in range(N):
            for j in range(D):
                binary[i] += x[i][j] * 2 ** j
        return torch.tensor(binary, dtype=torch.float32)



class nby2Transform(nn.Module):
    def __init__(self, vrange):
        super(nby2Transform, self).__init__()
        self.vrange = vrange
        self.bits = vrange // 2

    def forward(self, x):
        a = torch.zeros([self.bits], dtype=torch.float32)
        window = 2**self.bits - 1
        if x-self.bits >= 0:
            val = window >> x-self.bits
        else:
            val = window << self.bits-x
        
        val = val & window
        i = 0
        while (val > 0):
            mod = val % 2
            a[i] = mod
            val //= 2
            i += 1
        return a

class nby2ReverseTransform(nn.Module):
    def __init__(self, vrange):
        super(nby2ReverseTransform, self).__init__()
        self.vrange = vrange
        self.bits = vrange // 2
    
    def forward(self, x):
        N, D = x.shape
        x = (x.sign() + 1)/2

        num = torch.zeros([N], dtype=torch.float32)
        
        last = start = 0
        for i in range(N):
            nnz = torch.nonzero(x[i])
            if nnz.nelement() != 0:
                start = int(nnz[0])
                last = int(nnz[-1])
                num[i] = (self.bits - start) + (self.bits - last) - 1
        return num

# x = 2
# window = 0xf
# if x-4 >= 0:
#     val = window >> x-4
# else:
#     val = window << 4-x
# print(val)
# while (val > 0):
#     mod = val % 2
#     print(mod)
#     val //= 2


# tt = nby2Transform(20)
# trt = nby2ReverseTransform(20)

# for i in range(20):
#     f = tt(i)
#     print(f)
#     for i in range(len(f)):
#         if f[i] == 0:
#             f[i] = -1
#     f = trt(torch.tensor([f.numpy()]))

#     print(f)

# def gray_code(n):
#     def gray_code_recurse (g,n):
#         k=len(g)
#         if n<=0:
#             return

#         else:
#             for i in range (k-1,-1,-1):
#                 char='1'+g[i]
#                 g.append(char)
#             for i in range (k-1,-1,-1):
#                 g[i]='0'+g[i]

#             gray_code_recurse (g,n-1)

#     g=['0','1']
#     gray_code_recurse(g,n-1)
#     return g

# def main():

#     g = gray_code(7)
#     for i in range (len(g)):
#         # print('[', end='')
#         # for c in range(len(g[i]) - 1):
#         #     print(g[i][c], end=',')
#         # print(g[i][c], end='')
#         # print(']: ' + str(i), end='')
#         a = 0
#         for c in range(len(g[i])):
#             a |= int(g[i][c])
#             a = a << 1
#         a = a >> 1
        
#         print(str(a) + ':' + str(i), end=',')
        
# main()


class e2jmjTransform(nn.Module):
    def __init__(self, vrange):
        super(e2jmjTransform, self).__init__()
        self.bits = vrange // 4 + 2
        self.infile = pickle.load(open('/home/fzyxue/encodings/codings/e2jmj_' + str(vrange) + ".pkl", 'rb'))
        self.di = torch.transpose(self.infile, 0, 1)


    def forward(self, x):
        a = torch.zeros([self.bits], dtype=torch.float32)
        a = self.di[x]
        return a

class e2jmjReverseTransform(nn.Module):
    def __init__(self, vrange):
        super(e2jmjReverseTransform, self).__init__()
        self.vrange = vrange
        self.bits = vrange // 4 + 2
        self.exp = 2

    def forward(self, x):
        x = (x.sign() + 1)/2
        N, D = x.shape
        a = torch.zeros([N], dtype=torch.float32)
        for i in range(N):


            e = x[i, 0:self.exp]
            if torch.sum(e) == 0:
                e = 0
            else:
                e = self.exp // 2 - torch.nonzero(e)[-1] + self.exp // 2 - torch.nonzero(e)[0] + 1

            
            man = x[i, self.exp:]
            manrange = self.bits - self.exp

            if e % 2 == 0:
                if torch.sum(man) == 0:
                    man = 0
                else:
                    man = torch.nonzero(man)[-1]
            elif e % 2 == 1:
                if torch.sum(man) == 0:
                    man = 0
                else:
                    man = manrange - torch.nonzero(man)[-1] - 1

            a[i] = e * (self.bits - self.exp) + man

        return a


class hadamardTransform(nn.Module):
    def __init__(self, vrange):
        super(hadamardTransform, self).__init__()
        self.bits = vrange
        vrange = 2**int(np.ceil(np.log2(vrange)))
        self.infile = pickle.load(open('/home/fzyxue/encodings/codings/had_' + str(vrange) + ".pkl", 'rb'))
        self.di = torch.transpose(self.infile, 0, 1)

    def forward(self, x):
        a = torch.zeros([self.bits], dtype=torch.float32)
        a = self.di[x]
        return a

class hexjTransform(nn.Module):
    def __init__(self, vrange):
        super(hexjTransform, self).__init__()
        self.bits = int(np.ceil(np.log(vrange) / np.log(16)) * 9)
        vrange = 2**int(np.ceil(np.log2(vrange)))
        self.infile = pickle.load(open('/home/fzyxue/encodings/codings/hex_' + str(vrange) + ".pkl", 'rb'))
        self.di = torch.transpose(self.infile, 0, 1)

    def forward(self, x):
        a = torch.zeros([self.bits], dtype=torch.float32)
        a = self.di[x]
        return a

def match(encode, num_bits, di):
    encode = encode.float()
    t = torch.matmul(encode, di)

    return t

def match_decode(encode, num_bits, di):

    encode = encode.float()
    t = torch.matmul(encode, di)


    _,t = torch.max(t, dim=1)
    return t

def soft_match(encode, val_range, di):
    encode = encode.float()
    # encode = torch.tanh(encode)
    arr = torch.tensor(range(0, val_range, 1)).cuda()
    t = torch.matmul(encode, di)
    s = nn.Softmax(dim=1)
    t = s(t)
    t = t * arr

    return t

def soft_match_decode(encode, val_range, di):
    encode = encode.float()
    # encode = torch.tanh(encode)
    arr = torch.tensor(range(0, val_range, 1)).cuda()
    t = torch.matmul(encode, di)
    s = nn.Softmax(dim=1)
    t = s(t)
    t = t * arr
    ts = torch.sum(t, dim=1)

    return ts

class softCorrelationReverseTransform(nn.Module):
    def __init__(self, fwd):
        super(softCorrelationReverseTransform, self).__init__()
        self.bits = fwd.bits
        self.di = fwd.di
        self.val_range = self.di.shape[1]


    def forward(self, x):
        N, D = x.shape
        a = torch.zeros([N], dtype=torch.float32)
        for i in range(N):

            # print(x[i, :])
            # print(match_decode(x[i, :], self.bits, self.di))
            # exit()
            y = soft_match_decode(x[i, :].unsqueeze(0), self.val_range, self.di)
            a[i] = y

        return a
    
    def convert_discrete(self, x):
        N, D = x.shape
        a = torch.zeros((N, self.val_range), dtype=torch.float32).cuda()
        for i in range(N):
            # print(x[i, :])
            # print(match_decode(x[i, :], self.bits, self.di))
            # exit()
            y = soft_match(x[i, :].unsqueeze(0), self.val_range, self.di)
            a[i] = y
        return a

    def convert_continuous(self, x):
        N, D = x.shape
        a = torch.zeros((N), dtype=torch.float32).cuda()
        for i in range(N):
            # print(x[i, :])
            # print(match_decode(x[i, :], self.bits, self.di))
            # exit()
            y = soft_match_decode(x[i, :].unsqueeze(0), self.val_range, self.di)
            a[i] = y
        return a

class correlationReverseTransform(nn.Module):
    def __init__(self, fwd):
        super(correlationReverseTransform, self).__init__()
        self.bits = fwd.bits
        self.di = fwd.di
        self.val_range = self.di.shape[1]

    def forward(self, x):
        # x = (x.sign() + 1)/2
        N, D = x.shape
        a = torch.zeros([N], dtype=torch.float32)
        for i in range(N):
            # print(x[i, :])
            # print(match_decode(x[i, :], self.bits, self.di))
            # exit()
            y = match_decode(x[i, :].unsqueeze(0), self.val_range, self.di)
            a[i] = y

        return a

    def convert_discrete(self, x):
        N, D = x.shape
        a = torch.zeros((N, self.val_range), dtype=torch.float32).cuda()
        for i in range(N):
            # print(x[i, :])
            # print(match_decode(x[i, :], self.bits, self.di))
            # exit()
            y = match(x[i, :].unsqueeze(0), self.val_range, self.di)
            a[i] = y

        return a

    def convert_continuous(self, x):
        N, D = x.shape
        a = torch.zeros((N), dtype=torch.float32).cuda()
        for i in range(N):
            # print(x[i, :])
            # print(match_decode(x[i, :], self.bits, self.di))
            # exit()
            y = match_decode(x[i, :].unsqueeze(0), self.val_range, self.di)
            a[i] = y
        return a


class FileTransform(nn.Module):
    def __init__(self, filename):
        super(FileTransform, self).__init__()
        self.infile = pickle.load(open(filename, 'rb'))
        self.infile = torch.tensor(self.infile, dtype=torch.float32)
        self.bits = self.infile.shape[1]
        self.di = torch.transpose(self.infile, 0, 1).cuda()
        
    def forward(self, x):
        a = torch.zeros([self.bits], dtype=torch.float32)
        a = self.infile[x]
        return a

class e1jmjTransform(nn.Module):
    def __init__(self, vrange):
        super(e1jmjTransform, self).__init__()
        self.bits = vrange // 4 + 2
        self.infile = pickle.load(open('/home/fzyxue/encodings/codings/e1jmj_' + str(vrange) + ".pkl", 'rb'))
        self.di = torch.transpose(self.infile, 0, 1)

    def forward(self, x):
        a = torch.zeros([self.bits], dtype=torch.float32)
        a = self.di[x]
        return a

class e1jmjReverseTransform(nn.Module):
    def __init__(self, vrange):
        super(e1jmjReverseTransform, self).__init__()
        self.vrange = vrange
        self.bits = vrange // 2 + 1
        self.exp = 1

    def forward(self, x):
        x = (x.sign() + 1)/2
        N, D = x.shape
        a = torch.zeros([N], dtype=torch.float32)
        for i in range(N):


            e = x[i, 0:self.exp]
            if torch.sum(e) == 0:
                e = 0
            else:
                e = self.exp // 2 - torch.nonzero(e)[-1] + self.exp // 2 - torch.nonzero(e)[0] + 1

            
            man = x[i, self.exp:]
            manrange = self.bits - self.exp

            if e % 2 == 0:
                if torch.sum(man) == 0:
                    man = 0
                else:
                    man = torch.nonzero(man)[-1]
            elif e % 2 == 1:
                if torch.sum(man) == 0:
                    man = 0
                else:
                    man = manrange - torch.nonzero(man)[-1] - 1

            a[i] = e * (self.bits - self.exp) + man

        return a
        
# tt = e2jmjTransform(64)
# trt = correlationReverseTransform(tt)
# torch.set_printoptions(threshold=10000)
# print(tt.di)

# for i in range(64):
#     f = tt(i)
#     print(f)
#     for j in range(len(f)):
#         if f[j] == 0:
#             f[j] = -1
#     f = trt(torch.tensor([f.numpy()]))
#     print(i, f)
#     assert(i == f)