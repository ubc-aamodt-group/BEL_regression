import torch
import math
import sys

import pickle

BASE_DIR="~/BELJ_code/pilotnet/end2end-self-driving-car/encodings/"
BASE_DIR = "/home/fzyxue/encodings/codings/"

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

def match_decode(encode, num_bits, di):
    encode = encode.float()
    encode = torch.tanh(encode)
    
    t = torch.matmul(encode, di)

    _,t = torch.max(t, dim=0)
    
    return t

class none:
    def set_range(self,num_range,offset):
        self.num_range=num_range
        self.num_bits = 1
        self.offset=offset

    def encode(self, num):
        return num

    def decode(self, num):
        return num
        
class nby2:
    def set_range(self, num_range, offset):
        self.num_range=num_range
        self.num_bits = int(700/2) + 1
        self.offset=offset
        self.di = pickle.load(open(BASE_DIR + 'nby2_700.pkl', 'rb'))
        self.di = self.di.cuda()

    def encode(self, num):
        num=min(num,self.num_range)
        num=max(num,0)
        num = int(num)
        return self.di[:, num]

    def decode(self, encode):
        return match_decode(encode, self.num_bits, self.di)

class temp:
    def set_range(self, num_range, offset):
        self.num_range=num_range
        self.num_bits = 700
        self.offset=offset
        self.di = pickle.load(open(BASE_DIR + 'temp_700.pkl', 'rb'))
        self.di = self.di.cuda()

    def encode(self, num):
        num=min(num,self.num_range)
        num=max(num,0)
        num = int(num)
        return self.di[:, num]

    def decode(self, encode):
        return match_decode(encode, self.num_bits, self.di)

class e1jmj:
    def set_range(self, num_range, offset):
        self.num_range=num_range
        self.num_bits = int(1024 / 2) + 1
        self.offset=offset
        self.di = pickle.load(open(BASE_DIR + 'e1jmj_1024.pkl', 'rb'))
        self.di = self.di.cuda()

    def encode(self, num):
        num=min(num,self.num_range)
        num=max(num,0)
        num = int(num)
        return self.di[:, num]

    def decode(self, encode):
        return match_decode(encode, self.num_bits, self.di)

class e2jmj:
    def set_range(self, num_range, offset):
        self.num_range=num_range
        self.num_bits = int(1024 / 4) + 2
        self.offset=offset
        self.di = pickle.load(open(BASE_DIR + 'e2jmj_1024.pkl', 'rb'))
        self.di = self.di.cuda()

    def encode(self, num):
        num=min(num,self.num_range)
        num=max(num,0)
        num = int(num)
        return self.di[:, num]

    def decode(self, encode):

        return match_decode(encode, self.num_bits, self.di)

class hexj:
    def set_range(self, num_range, offset):
        self.num_range=num_range
        self.num_bits = int(np.ceil(np.log(1024) / np.log(16)) * 9)
        self.offset=offset
        self.di = pickle.load(open(BASE_DIR + 'hex_1024.pkl', 'rb'))
        self.di = self.di.cuda()

    def encode(self, num):
        num=min(num,self.num_range)
        num=max(num,0)
        num = int(num)
        return self.di[:, num]

    def decode(self, encode):
        return match_decode(encode, self.num_bits, self.di)

class had:
    def set_range(self, num_range, offset):
        self.num_range=num_range
        self.num_bits = 1024
        self.offset=offset
        self.di = pickle.load(open(BASE_DIR + 'had_1024.pkl', 'rb'))
        self.di = self.di.cuda()

    def encode(self, num):
        num=min(num,self.num_range)
        num=max(num,0)
        num = int(num)
        return self.di[:, num]

    def decode(self, encode):
        return match_decode(encode, self.num_bits, self.di)
    
class nby2vert:
    def set_range(self,num_range,offset):
        self.num_range=num_range
        self.num_bits = int(num_range/2)
        self.offset=offset

    def encode(self,num):
        num=min(num,self.num_range-1)
        num=max(num,0)
        bits= self.num_bits
        a= torch.zeros([bits],dtype=torch.long)
        for i in range(0,bits):
            if num > (bits-i-1) and num <= self.num_range-i-1:
                a[i] =1
        return a

    def decode(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits):
                if code[j]>0:
                    start=num_bits-j
                    break
            for j in range(num_bits-1,-1,-1):
                if code[j]>0:
                    last = num_bits-1-j
                    break
            number[index]=float(start+last)
            index+=1
        return number

    def decode_2bits(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits):
                if code[j]>0:
                    start=num_bits-j
                    break
            for j in range(num_bits-1,-1,-1):
                if code[j]>0:
                    last = num_bits-1-j
                    break
            number[index]=float(start+last)
            index+=1
        return number


import numpy as np 
class temperature:
    def set_range(self,num_range,offset):
        self.num_range=num_range
        self.num_bits = int(num_range)+1
        self.offset=offset

    def encode(self,num):
        num=int(np.round(num))
        num=min(num,self.num_range-1)
        num=max(num,0)
        bits= self.num_bits
        a= torch.zeros([bits],dtype=torch.long)
        for i in range(bits-1,bits-3-num,-1):
            a[i]=1
        return a

    def decode(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits-1):
                if code[j]>0:
                    last = num_bits-2-j
                    break
            number[index]=float(last)
            index+=1
        return number

    def decode_2bits(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits-1):
                if code[j]>0 and code[j+1]>0:
                    last = num_bits-2-j
                    break
            number[index]=float(last)
            index+=1
        return number

class onehot:
    def set_range(self,num_range,offset):
        self.num_range=num_range
        self.num_bits = int(num_range)
        self.offset=offset

    def encode(self,num):
        num=int(round(num))
        num=min(num,self.num_range-1)
        num=max(num,0)
        bits= self.num_bits
        a= torch.zeros([bits],dtype=torch.long)
        for i in range(bits-1-num,bits-2-num,-1):
            a[i]=1
        return a

    def decode(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits):
                if code[j]>0:
                    last = num_bits-1-j
                    break
            number[index]=float(last)
            index+=1
        return number

    def decode_2bits(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits):
                if code[j]>0:
                    last = num_bits-1-j
                    break
            number[index]=float(last)
            index+=1
        return number

class twohot:
    def set_range(self,num_range,offset):
        self.num_range=num_range
        self.num_bits = int(num_range)+1
        self.offset=offset
    def encode(self,num):
        num=int(round(num))
        num=min(num,self.num_range-1)
        num=max(num,0)
        bits= self.num_bits
        a= torch.zeros([bits],dtype=torch.long)
        for i in range(bits-1-num,bits-3-num,-1):
            a[i]=1
        return a

    def decode(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits-1):
                if code[j]>0:
                    last = num_bits-2-j
                    break
            number[index]=float(last)
            index+=1
        return number

    def decode_2bits(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits-1):
                if code[j]>0 and code[j+1]:
                    last = num_bits-2-j
                    break
            number[index]=float(last)
            index+=1
        return number

class fourhot:
    def set_range(self,num_range,offset):
        self.num_range=num_range
        self.num_bits = int(num_range)+3
        self.offset=offset

    def encode(self,num):
        num=int(round(num))
        num=min(num,self.num_range-1)
        num=max(num,0)
        bits= self.num_bits
        a= torch.zeros([bits],dtype=torch.long)
        for i in range(bits-1-num,bits-5-num,-1):
            a[i]=1
        return a

    def decode(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits-1):
                if code[j]>0:
                    last = num_bits-4-j
                    break
            number[index]=float(last)
            index+=1
        return number

    def decode_2bits(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits-1):
                if code[j]>0 and code[j+1]:
                    last = num_bits-4-j
                    break
            for j in range(num_bits-1,0,-1):
                if code[j]>0 and code[j-1]:
                    start = num_bits-1-j
                    break
            number[index]=float((last+start)/2)
            index+=1
        return number

class nby2hor:
    def set_range(self,num_range,offset):
        self.num_range=num_range
        self.num_bits = int(num_range) + int(num_range/2) -1
        self.offset=offset

    def encode(self,num):
        num=int(round(num))
        num=min(num,self.num_range-1)
        num=max(num,0)
        bits= self.num_bits
        a= torch.zeros([bits],dtype=torch.long)
        for i in range(bits-1-num,bits-1-num-int(self.num_range/2),-1):
            a[i]=1
        return a

    def decode(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits-1):
                if code[j]>0:
                    last = num_bits-j-int(self.num_range/2)
                    break
            number[index]=float(last)
            index+=1
        return number

    def decode_2bits(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        start =0
        last = 0
        index=0
        for code in encode:
            for j in range(0,num_bits-1):
                if code[j]>0 and code[j+1]>0:
                    last = num_bits-j-1#-int(self.num_range/2)
                    break
            for j in range(num_bits-1,0,-1):
                if code[j]>0 and code[j-1]:
                    start = num_bits-j-int(self.num_range/2)
                    break
            number[index]=float((last+start)/2)
            index+=1
        return number

class bcd:
    def set_range(self,num_range,offset):
        self.num_range=num_range
        self.num_bits =  math.ceil(math.log2(num_range))
        self.offset=offset
    def encode(self,num):
        num=int(np.round(num))
        num=min(num,self.num_range-1)
        num=max(num,0)
        bits= self.num_bits
        a= torch.zeros([bits],dtype=torch.long)
        for i in range(bits):
            if num%2==0:
                a[i]=0
            else:
                a[i]=1
            num=int(num/2)
        return a

    def decode(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        index=0
        for code in encode:
            a=0
            mult=1
            for j in range(0,num_bits):
                a+=code[j]*mult
                mult=mult*2
            number[index]=float(a)
            index+=1
        return number

    def decode_2bits(self,encode):
        num_bits=self.num_bits
        number= torch.zeros([encode.size(0)])
        index=0
        for code in encode:
            a=0
            mult=1
            for j in range(0,num_bits):
                a+=code[j]*mult
                mult=mult*2
            number[index]=float(a)
            index+=1
        return number

def verify(converter,test):
    converter.set_range(test,0)
    a=torch.ones([test,converter.num_bits])
    for i in range(0,test):
        print(i)
        print(converter.encode(i))
        a[i]= converter.encode(i)
    print(a)
    print(converter.decode(a))
    print(converter.decode_2bits(a))


