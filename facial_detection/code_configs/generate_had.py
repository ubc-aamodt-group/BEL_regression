import torch
from scipy.linalg import hadamard
def pt(a,num):
	a=a.numpy()
	for i in a:
		print(i, end="")
	print(" ",num)

import pickle
di={}
encode=torch.zeros((256,256))
arr= hadamard(256)
x = torch.Tensor(list(arr))
x= (x+1/2).long()
print(x)
for i in range(0,256):
	#print(temp_code(i,256),i)
	pt(x[i],i)
	di[i]=x[i]
	encode[i]=x[i]
##check=convert(encode,64,arr,arrs,finmult)
f=open("had_256_code.pkl","wb")
pickle.dump(di,f)
f=open("had_256_tensor.pkl","wb")
pickle.dump(encode,f)
