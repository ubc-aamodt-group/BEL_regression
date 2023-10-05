import torch
def temp_code(num,num_bits):
    num=max(0,num)
    bits= int(num_bits/2)
    a= torch.zeros([bits],dtype=torch.long)
    for i in range(0,bits):
        if num > (bits-i-1) and num <= num_bits-i-1:
            a[i] =1

    return a

import pickle

di={}
for i in range(0,256):
	print(temp_code(i,256),256)
	di[i]=temp_code(i,256)
f=open("nby2_256_code.pkl","wb")
pickle.dump(di,f)
