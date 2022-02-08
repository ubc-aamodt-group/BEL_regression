# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:44:09 2018

@author: beheraa

from custom_validate_callback import TestCallback

callbacks = [TestCallback(test_datagen, 10)] # the model evaluate every 10 epochs
"""
import keras
from sklearn.metrics import accuracy_score
import numpy as np
import csv
from os.path import dirname, realpath
import math
import tensorflow as tf
import scipy
def convert_new_tanh(encode,di): # based on correlation
    #t = np.matmul(encode,di)
    ts = np.argmax(encode,axis=1)
    return (ts)
def convert_new_tanh_verify(encode,di): # based on correlation
    #print(encode.shape)
    #print(di)
    #encode = (encode*2)-1
    #t = np.matmul(encode,di)
    ts = np.argmax(encode,axis=1)
    return (ts)
def convert_new_tanh_loss(encode,di): # based on correlation
    t = np.matmul(encode,di)
    return (t)

def convert(encode,di):
    arr=np.array(range(0,360,1))
    #encode = np.tanh(encode)
    #t = np.matmul(encode,di)
    t = scipy.special.softmax(encode, axis=1)
    t = t * arr
    ts = np.sum(t,axis=1)
    return (ts)

def measure_MAE(label_yaw,label_pitch,label_roll,yaw,pitch,roll,batch_size,di,code_name="belj"):
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
            '''
            try:
                loss, acc = self.model.evaluate_generator(self.test_generator)
                #prediction = self.model.predict_generator(self.test_generator)
            except:
                #model is regression, so must approximate accuracy
                loss = self.model.evaluate_generator(self.test_generator)
                acc = validateRegression(self.test_generator, self.model)
                
            print('\nValidation loss: {}, acc: {}\n'.format(loss, acc))
            '''
            acc = validateRegression(self.test_generator, self.model,self.code_tensor,self.code_name)
            if acc[3]<self.best_acc:
                 self.best_acc=acc[3]
                 print("best accuracy %s achieved"%(str(acc)))
                 self.model.save_weights(self.metrics_dir+"/best_model")                 
           
            val_out = self.model.evaluate_generator(self.test_generator)
            print('\nValidation loss: {}, acc: {}\n'.format(val_out, acc))
            writeValToCSV(self, epoch, val_out+acc, self.metrics_dir)
                
                
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
        y,p,r=measure_MAE(y_true[0],y_true[1],y_true[2],pred[3],pred[4],pred[5],batch_size,di,code_name=code_name)
        y1,p1,r1=measure_MAE1(y_true[0],y_true[1],y_true[2],pred[3],pred[4],pred[5],batch_size,di)
        y2,p2,r2=measure_MAE2(y_true[0],y_true[1],y_true[2],pred[3],pred[4],pred[5],batch_size,di  )
        yc,pc,rc=measure_MAE2v(y_true[0],y_true[1],y_true[2],y_true[3],y_true[4],y_true[5],batch_size,di)
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
    
    #print(predsAcc)
    #print(trues)
    ye = ye*180/(t*np.pi); pe = pe*180/(t*np.pi); re = re*180/(t*np.pi)
    print("validation_o yaw pitch roll MAE"); print(ye,pe,re,(ye+pe+re)/3)
    ye1 = ye1*180/(t*np.pi); pe1 = pe1*180/(t*np.pi); re1 = re1*180/(t*np.pi)
    print("validation_genex yaw pitch roll MAE"); print(ye1,pe1,re1,(ye1+pe1+re1)/3)
    ye2 = ye2*180/(t*np.pi); pe2 = pe2*180/(t*np.pi); re2 = re2*180/(t*np.pi)
    print("validation_gen yaw pitch roll MAE"); print(ye2,pe2,re2,(ye2+pe2+re2)/3)
    yce = yce*180/(t*np.pi); pce = pce*180/(t*np.pi); rce = rce*180/(t*np.pi)
    print("confrim validation yaw pitch roll MAE"); print(yce,pce,rce,(yce+pce+rce)/3)
    return [ye,pe,re,(ye+pe+re)/3,ye1,pe1,re1,(ye1+pe1+re1)/3,ye2,pe2,re2,(ye2+pe2+re2)/3]

#writes validation metrics to csv file
def writeValToCSV(self, epoch, val_out,metrics_dir):
    
    #get root directory
    filepath = realpath(__file__)
    #metrics_dir = dirname(dirname(filepath)) + '/Metrics/'
    
    
    with open(metrics_dir + "/"+ self.model_name + '(Validation).csv', 'a', newline='') as csvFile:
        metricWriter = csv.writer(csvFile)
        row = [epoch] + val_out
        metricWriter.writerow(row)
        #metricWriter.writerow([epoch, loss, acc])
