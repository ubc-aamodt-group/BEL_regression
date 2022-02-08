import keras
from sklearn.metrics import accuracy_score
import numpy as np
import csv
from os.path import dirname, realpath
import math
import tensorflow as tf
import scipy

def convert_new_tanh(encode,di): # based on correlation
    #encode = np.tanh(encode)
    t = np.matmul(encode,di)
    ts = np.argmax(t,axis=1)
    return (ts)
def convert_new_tanh_verify(encode,di): # based on correlation
    encode = (encode*2)-1
    t = np.matmul(encode,di)
    ts = np.argmax(t,axis=1)
    return (ts)
def convert_new_tanh_loss(encode,di): # loss calculation based on correlation
    #encode = np.tanh(encode)
    t = np.matmul(encode,di)
    return (t)

def convert(encode,di):
    arr=np.array(range(0,360,1))
    #encode = np.tanh(encode)
    t = np.matmul(encode,di)
    t = scipy.special.softmax(t, axis=1)
    t = t * arr
    ts = np.sum(t,axis=1)
    return (ts)

def convert_belj(encode,num_bits):
    number= np.zeros(((encode.shape)[0]))
    arr=np.array(range(num_bits,0,-1))
    arrs = np.array(range(1,num_bits+1,1))
    encode = (np.sign(encode)+1)/2
    temp=encode*arr
    ts = np.max(temp,axis=1)
    temp=encode*arrs
    temp = np.max(temp,axis=1)
    te=180-temp
    return (ts+te)

def convert_belu(encode,num_bits):
    encode = (np.sign(encode)+1)/2
    number=np.sum(encode,axis=1)
    return number

#Decode BEU or BELJ or GEN
def measure_MAE(label_yaw,label_pitch,label_roll,yaw,pitch,roll,batch_size,di,code_name="belj"):
    if code_name=="belj" or code_name=="belj3":
	    yaw_predicted=((convert_belj(yaw,180)-180)*math.pi)/180
	    pitch_predicted= ((convert_belj(pitch,180)-180)*math.pi)/180
	    roll_predicted=((convert_belj(roll,180)-180)*math.pi)/180
    elif code_name=="belu":
	    yaw_predicted=((convert_belu(yaw,180)-180)*math.pi)/180
	    pitch_predicted= ((convert_belu(pitch,180)-180)*math.pi)/180
	    roll_predicted=((convert_belu(roll,180)-180)*math.pi)/180
    else:
	    yaw_predicted=((convert_new_tanh(yaw,di)-180)*math.pi)/180
	    pitch_predicted= ((convert_new_tanh(pitch,di)-180)*math.pi)/180
	    roll_predicted=((convert_new_tanh(roll,di)-180)*math.pi)/180
    label_yaw = np.reshape(label_yaw, (batch_size))
    label_pitch = np.reshape(label_pitch, (batch_size))
    label_roll = np.reshape(label_roll, (batch_size))
    yaw_error = np.sum(np.abs(yaw_predicted - label_yaw))
    pitch_error = np.sum(np.abs(pitch_predicted - label_pitch))
    roll_error = np.sum(np.abs(roll_predicted - label_roll))
    return yaw_error, pitch_error,roll_error
                                     
# decode Gen-EX
def measure_MAE1(label_yaw,label_pitch,label_roll,yaw,pitch,roll,batch_size,di):
    yaw_predicted=((convert(yaw,di)-180)*math.pi)/180
    pitch_predicted= ((convert(pitch,di)-180)*math.pi)/180
    roll_predicted=((convert(roll,di)-180)*math.pi)/180
    label_yaw = np.reshape(label_yaw, (batch_size))
    label_pitch = np.reshape(label_pitch, (batch_size))
    label_roll = np.reshape(label_roll, (batch_size))
    yaw_error = np.sum(np.abs(yaw_predicted - label_yaw))
    pitch_error = np.sum(np.abs(pitch_predicted - label_pitch))
    roll_error = np.sum(np.abs(roll_predicted - label_roll))
    return yaw_error, pitch_error,roll_error
    
#decode GEN
def measure_MAE2(label_yaw,label_pitch,label_roll,yaw,pitch,roll,batch_size,di):
    yaw_predicted=((convert_new_tanh(yaw,di)-180)*math.pi)/180
    pitch_predicted= ((convert_new_tanh(pitch,di)-180)*math.pi)/180
    roll_predicted=((convert_new_tanh(roll,di)-180)*math.pi)/180
    label_yaw = np.reshape(label_yaw, (batch_size))
    label_pitch = np.reshape(label_pitch, (batch_size))
    label_roll = np.reshape(label_roll, (batch_size))
    yaw_error = np.sum(np.abs(yaw_predicted - label_yaw))
    pitch_error = np.sum(np.abs(pitch_predicted - label_pitch))
    roll_error = np.sum(np.abs(roll_predicted - label_roll))
    return yaw_error, pitch_error,roll_error

#Quantization error
def measure_MAE2v(label_yaw,label_pitch,label_roll,yaw,pitch,roll,batch_size,di):
    yaw_predicted=((convert_new_tanh_verify(yaw,di)-180)*math.pi)/180
    pitch_predicted= ((convert_new_tanh_verify(pitch,di)-180)*math.pi)/180
    roll_predicted=((convert_new_tanh_verify(roll,di)-180)*math.pi)/180
    label_yaw = np.reshape(label_yaw, (batch_size))
    label_pitch = np.reshape(label_pitch, (batch_size))
    label_roll = np.reshape(label_roll, (batch_size))
    yaw_error = np.sum(np.abs(yaw_predicted - label_yaw))
    pitch_error = np.sum(np.abs(pitch_predicted - label_pitch))
    roll_error = np.sum(np.abs(roll_predicted - label_roll))
    return yaw_error, pitch_error,roll_error
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, test_generator, test_steps, model_name,code_tensor="codes/belj_360_tensor.pkl",metrics_dir="Metrics/",code_name="belj"):
        self.test_generator = test_generator
        self.test_steps = test_steps
        self.model_name = model_name
        self.code_tensor=code_tensor
        self.metrics_dir=metrics_dir
        self.code_name=code_name
        self.best_acc=100
    def on_epoch_end(self, epoch, logs={}):        
        if (epoch + 1) % self.test_steps == 0:# and epoch != 0:
            acc = validateRegression(self.test_generator, self.model,self.code_tensor,self.code_name)
            print('\n acc: {}\n'.format(acc))
                
                
import pickle
def validateRegression(val_dg, model,code_tensor,code_name):
    predsAcc=[]
    trues=[]
    ye,pe,re,t=0.0,0.0,0.0,0
    ye1,pe1,re1=0.0,0.0,0.0
    ye2,pe2,re2=0.0,0.0,0.0
    yce,pce,rce,t=0.0,0.0,0.0,0
    pitcherror=0.0
    di = pickle.load(open(code_tensor,"rb"))
    di = np.transpose(di).astype('float64')
    for b in range(len(val_dg)):
        x, y_true = val_dg.__getitem__(b)
        pred = model.predict(x)
        #print(pred[0].shape,pred[1].shape,pred[2].shape,pred[3].shape,pred[4].shape,pred[5].shape)
        batch_size = (pred[0].shape)[0]
        r,p,y=measure_MAE(y_true[0],y_true[1],y_true[2],pred[3],pred[4],pred[5],batch_size,di,code_name=code_name)
        r1,p1,y1=measure_MAE1(y_true[0],y_true[1],y_true[2],pred[3],pred[4],pred[5],batch_size,di)
        r2,p2,y2=measure_MAE2(y_true[0],y_true[1],y_true[2],pred[3],pred[4],pred[5],batch_size,di  )
        rc,pc,yc=measure_MAE2v(y_true[0],y_true[1],y_true[2],y_true[3],y_true[4],y_true[5],batch_size,di)
        #print("per batch total ypr")
        #print(y,p,r)
        t+=batch_size
        ye+=y
        pe+=p
        re+=r
        ye1+=y1
        pe1+=p1
        re1+=r1
        ye2+=y2
        pe2+=p2
        re2+=r2
        yce+=yc
        pce+=pc
        rce+=rc
    
    ye = ye*180/(t*np.pi); pe = pe*180/(t*np.pi); re = re*180/(t*np.pi)
    print("validation_belu/belj/gen yaw: %f, pitch: %f, roll: %f, MAE: %f"%(ye,pe,re,(ye+pe+re)/3))
    ye1 = ye1*180/(t*np.pi); pe1 = pe1*180/(t*np.pi); re1 = re1*180/(t*np.pi)
    print("validation_genex yaw: %f, pitch: %f, roll: %f, MAE: %f"%(ye1,pe1,re1,(ye1+pe1+re1)/3))
    ye2 = ye2*180/(t*np.pi); pe2 = pe2*180/(t*np.pi); re2 = re2*180/(t*np.pi)
    print("validation_gen yaw: %f, pitch: %f, roll: %f, MAE: %f"%(ye2,pe2,re2,(ye2+pe2+re2)/3))
    yce = yce*180/(t*np.pi); pce = pce*180/(t*np.pi); rce = rce*180/(t*np.pi)
    print("confrim validation yaw: %f, pitch: %f, roll: %f, MAE: %f"%(yce,pce,rce,(yce+pce+rce)/3))
    return [ye,pe,re,(ye+pe+re)/3,ye1,pe1,re1,(ye1+pe1+re1)/3,ye2,pe2,re2,(ye2+pe2+re2)/3]
