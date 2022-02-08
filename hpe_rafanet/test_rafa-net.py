import time

import numpy as np
import cv2, os, math
import sys
from keras import layers
import tensorflow as tf
from keras import backend as K
from keras.models import Model
import keras
import keras.models as load
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from custom_validate_callback import CustomCallback

from keras.applications.resnet50 import preprocess_input as res50_pp_input
from pose_data_augmentor import DirectoryDataGenerator
from custom_validate_callback import CustomCallback
from custom_validate_callback import validateRegression
from keras.applications.resnet50 import ResNet50
from SelfAttention import SelfAttention
from SpectralNormalizationKeras import ConvSN2D
#from keras_self_attention import SeqWeightedAttention as Attention
#from keras_self_attention import SeqSelfAttention 

from attentional_spatial_pooling import WeightedAttention as Attention
from attentional_spatial_pooling import AttentionalPooling as SeqSelfAttention


working_dir = os.path.dirname(os.path.realpath(__file__))
model_name = "{}_bs{}".format(sys.argv[0].split(".")[0], 16)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def resnet_block5_weights(model, tensor_b1, base_layer):
    #tensor_b1 = layers.Input(shape=(14, 14, 1024))
    #base_out = layers.Input(shape=(14, 14, 1024))
    x = conv_block(tensor_b1, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    ''' The output of x (7,7,2048). Add a pooling layer: experiment with global 
    pooling and local pooling with (max and average option) with flatten in the 
    end to feed to Attention
    '''
    #x = layers.GlobalMaxPooling2D()(x)
    res_block5 = Model(inputs = tensor_b1, outputs = x)
    #res_block5.summary()
    
    ''' Set the weights from the original network '''
    for i in range(1, 33): #extra pooling layer in res_block5
        j = i + base_layer
        weights = model.layers[j].get_weights()
        print("source: \t", model.layers[j].name)
        print("target: \t", res_block5.layers[i].name)
        if len(weights) == 0:
            continue
        res_block5.layers[i].set_weights(weights)
        #print(res_block5.layers[i].name)
    return res_block5

def load_image_files_and_labels(image_path, label_path, radian=True):
    img_list = open(image_path).readlines()	#'./AFLW_meta.tsv'
    label_list = open(label_path).readlines()
    files = []
    labels = []
    Bbox = []
    
    for index in range(len(img_list)):
        #if(index % 5000 == 0):
        #    print(str(index) + "	Hit!")
        line_image = img_list[index]
        line_label = label_list[index]
        
        items = line_image.split('\t')
        if (len(items) != 2 or items[1]==' \n' or items[1]=='\n'):
            #print(index, items[0], len(items))
            continue
        #print(items[0])
        '''
        try:
            x_set = int(items[1].split(' ')[1])
            y_set = int(items[1].split(' ')[0])
        except Exception as e:
            print(items[0], items[1])
        '''
        x_set = int(items[1].split(' ')[1])
        y_set = int(items[1].split(' ')[0])
        
        if (x_set<0 or y_set<0):
            #print(index, items[0], x_set, y_set)
            continue
        
        files.append(items[0][1:])
        offset = int(float(items[1].split(' ')[2]))
        Bbox.append(np.array([x_set, y_set, offset]))
        items_label = line_label.split('\t')
        if radian:            
            R_Angle = np.float32(items_label[1])
            P_Angle = np.float32(items_label[2])
            Y_Angel = np.float32(items_label[3])   
        else: #degrees (-180 to +180)
            R_Angle = np.float32(items_label[1])* math.pi /180
            P_Angle = np.float32(items_label[2])* math.pi /180
            Y_Angel = np.float32(items_label[3])* math.pi /180
            
        labels.append(np.array([R_Angle, P_Angle, Y_Angel]))
        
    return files, labels, Bbox
        

if __name__ == '__main__':
    theta= int(sys.argv[1])
    code_pkl= sys.argv[2] 
    code_tensor= sys.argv[3]
    code_bits= int(sys.argv[4])
    code_name= sys.argv[5]
    gamma= float(sys.argv[6])
    model_filename = sys.argv[7]
    train_image = '300wcrop.txt'	
    train_label = '300w_euler_cls.txt'
    test_image = 'Aflw2000crop.txt'
    test_label = 'Aflw2000_euler_cls.txt'

    
    train_files, train_labels, train_Bbox = load_image_files_and_labels(train_image, train_label)
    test_files, test_labels, test_Bbox = load_image_files_and_labels(test_image, test_label)
    

    input_tensor = layers.Input(shape=(224, 224, 3))
    nb_classes = 5 #number of image classes
    

    image_size = (224,224) #image resolution in pixels
    batch_size = 16 #number of images to process in one go
    epochs = 100 #number of times to pass whole training dataset though the network
    nb_train_samples =   len(train_files) #number of images used for training
    nb_test_samples = 0 #number of images used for testing
    nb_val_samples = len(test_files) #number of images used for validation
    verbose = 10
    validation_steps = 1 #number of epochs between validation
    optimizer = RMSprop()

    #loss_type = 'mean_squared_error'
    metrics = ['mae']

    '''Just before the block5 to capture the output'''
    base_layer = 142
    
    model = ResNet50(weights='imagenet', input_tensor=input_tensor, include_top=False)
    
    
    base_out = model.layers[base_layer].output #tapped output before block5
    #x = SelfAttentionGoogleBrain(filters=1024)(base_out)
    #y = SelfAttentionGoogleBrain(filters=1024)(base_out)
    #z = SelfAttentionGoogleBrain(filters=1024)(base_out)
    '''
    x_f = ConvSN2D(128, kernel_size=1, strides=1, padding='same')(base_out)# [bs, h, w, c']
    x_g = ConvSN2D(128, kernel_size=1, strides=1, padding='same')(base_out) # [bs, h, w, c']
    x_h = ConvSN2D(1024, kernel_size=1, strides=1, padding='same')(base_out)
    
    y_f = ConvSN2D(128, kernel_size=1, strides=1, padding='same')(base_out)# [bs, h, w, c']
    y_g = ConvSN2D(128, kernel_size=1, strides=1, padding='same')(base_out) # [bs, h, w, c']
    y_h = ConvSN2D(1024, kernel_size=1, strides=1, padding='same')(base_out)
    
    z_f = ConvSN2D(128, kernel_size=1, strides=1, padding='same')(base_out)# [bs, h, w, c']
    z_g = ConvSN2D(128, kernel_size=1, strides=1, padding='same')(base_out) # [bs, h, w, c']
    z_h = ConvSN2D(1024, kernel_size=1, strides=1, padding='same')(base_out)
    
    x = SelfAttention(filters=1024)([base_out, x_f, x_g, x_h])
    y = SelfAttention(filters=1024)([base_out, y_f, y_g, y_h])
    z = SelfAttention(filters=1024)([base_out, z_f, z_g, z_h])
    '''
    tensor_yaw = layers.Input(shape=(14, 14, 1024)) #Input to the parallal stream
    tensor_pitch = layers.Input(shape=(14, 14, 1024)) #Input to the parallal stream
    tensor_roll = layers.Input(shape=(14, 14, 1024)) #Input to the parallal stream
    yaw = resnet_block5_weights(model,tensor_yaw, base_layer)
    pitch = resnet_block5_weights(model,tensor_pitch, base_layer)
    roll = resnet_block5_weights(model,tensor_roll, base_layer)
    
    x = yaw(base_out)
    x_f = ConvSN2D(256, kernel_size=1, strides=1, padding='same')(x)# [bs, h, w, c']
    x_g = ConvSN2D(256, kernel_size=1, strides=1, padding='same')(x) # [bs, h, w, c']
    x_h = ConvSN2D(2048, kernel_size=1, strides=1, padding='same')(x)
    x = SelfAttention(filters=2048)([x, x_f, x_g, x_h])

    p1 = layers.MaxPooling2D(2, strides = 2, padding='valid')(x)
    p1 = layers.Reshape((-1,2048))(p1)
    p2 = layers.MaxPooling2D(3, strides = 2, padding='valid')(x)
    p2 = layers.Reshape((-1,2048))(p2)
    p3 = layers.MaxPooling2D(3, strides = 3, padding='valid')(x)
    p3 = layers.Reshape((-1,2048))(p3)
    p4 = layers.MaxPooling2D(4, strides = 2, padding='valid')(x)
    p4 = layers.Reshape((-1,2048))(p4)
    p5 = layers.MaxPooling2D(4, strides = 3, padding='valid')(x)
    p5 = layers.Reshape((-1,2048))(p5)
    p6 = layers.MaxPooling2D(5, strides = 2, padding='valid')(x)
    p6 = layers.Reshape((-1,2048))(p6)
    
    #x1 = layers.GlobalAveragePooling2D(name='avg_pool_yaw')(x)
    x2 = layers.GlobalMaxPooling2D(name='max_pool_yaw')(x)
    #x1 = layers.Reshape((1,2048))(x1)
    x2 = layers.Reshape((1,2048))(x2)
    x = layers.concatenate([p1, p2, p3, p4, p5, p6,x2], axis = 1)
    #x = layers.concatenate([x,x2], axis = 1)
    x = SeqSelfAttention(name='pooling_attn_yaw')(x)
    x = Attention(name='attention_yaw')(x)
    x = layers.BatchNormalization()(x)
    x1 = layers.Dense(theta, activation='linear', name="Yaw_int")(x)
    xb = layers.Dense(code_bits, activation='linear', name="Yaw_outputb")(x1)
    x = layers.Dense(1, activation='linear', name="Yaw_output")(x)
    
    y = pitch(base_out)
    y_f = ConvSN2D(256, kernel_size=1, strides=1, padding='same')(y)# [bs, h, w, c']
    y_g = ConvSN2D(256, kernel_size=1, strides=1, padding='same')(y) # [bs, h, w, c']
    y_h = ConvSN2D(2048, kernel_size=1, strides=1, padding='same')(y)
    y = SelfAttention(filters=2048)([y, y_f, y_g, y_h])
    
    p1 = layers.MaxPooling2D(2, strides = 2, padding='valid')(y)
    p1 = layers.Reshape((-1,2048))(p1)
    p2 = layers.MaxPooling2D(3, strides = 2, padding='valid')(y)
    p2 = layers.Reshape((-1,2048))(p2)
    p3 = layers.MaxPooling2D(3, strides = 3, padding='valid')(y)
    p3 = layers.Reshape((-1,2048))(p3)
    p4 = layers.MaxPooling2D(4, strides = 2, padding='valid')(y)
    p4 = layers.Reshape((-1,2048))(p4)
    p5 = layers.MaxPooling2D(4, strides = 3, padding='valid')(y)
    p5 = layers.Reshape((-1,2048))(p5)
    p6 = layers.MaxPooling2D(5, strides = 2, padding='valid')(y)
    p6 = layers.Reshape((-1,2048))(p6)
    
    #y1 = layers.GlobalAveragePooling2D(name='avg_pool_pitch')(y)
    y2 = layers.GlobalMaxPooling2D(name='max_pool_pitch')(y)
    #y1 = layers.Reshape((1,2048))(y1)
    y2 = layers.Reshape((1,2048))(y2)
    #y = layers.concatenate([y1,y2], axis = 1)
    y = layers.concatenate([p1, p2, p3, p4, p5, p6, y2], axis = 1)
    y = SeqSelfAttention(name='pooling_attn_pitch')(y)
    y = Attention(name='attention_pitch')(y)
    y = layers.BatchNormalization()(y)
    y1 = layers.Dense(theta, activation='linear', name="Pitch_int")(y)
    yb = layers.Dense(code_bits, activation='linear', name="Pitch_outputb")(y1)
    y = layers.Dense(1, activation='linear', name="Pitch_output")(y)
    
    z = roll(base_out)
    z_f = ConvSN2D(256, kernel_size=1, strides=1, padding='same')(z)# [bs, h, w, c']
    z_g = ConvSN2D(256, kernel_size=1, strides=1, padding='same')(z) # [bs, h, w, c']
    z_h = ConvSN2D(2048, kernel_size=1, strides=1, padding='same')(z)
    z = SelfAttention(filters=2048)([z, z_f, z_g, z_h])
    
    p1 = layers.MaxPooling2D(2, strides = 2, padding='valid')(z)
    p1 = layers.Reshape((-1,2048))(p1)
    p2 = layers.MaxPooling2D(3, strides = 2, padding='valid')(z)
    p2 = layers.Reshape((-1,2048))(p2)
    p3 = layers.MaxPooling2D(3, strides = 3, padding='valid')(z)
    p3 = layers.Reshape((-1,2048))(p3)
    p4 = layers.MaxPooling2D(4, strides = 2, padding='valid')(z)
    p4 = layers.Reshape((-1,2048))(p4)
    p5 = layers.MaxPooling2D(4, strides = 3, padding='valid')(z)
    p5 = layers.Reshape((-1,2048))(p5)
    p6 = layers.MaxPooling2D(5, strides = 2, padding='valid')(z)
    p6 = layers.Reshape((-1,2048))(p6)
    
    
    #z1 = layers.GlobalAveragePooling2D(name='avg_pool_roll')(z)
    z2 = layers.GlobalMaxPooling2D(name='max_pool_roll')(z)
    #z1 = layers.Reshape((1,2048))(z1)
    z2 = layers.Reshape((1,2048))(z2)
    z = layers.concatenate([p1, p2, p3, p4, p5, p6, z2], axis = 1)
    z = SeqSelfAttention(name='pooling_attn_roll')(z)
    
    #z = layers.concatenate([z1,z2], axis = 1)
    z = Attention(name='attention_roll')(z)
    z = layers.BatchNormalization()(z)
    z1 = layers.Dense(theta, activation='linear', name="Roll_int")(z)
    zb = layers.Dense(code_bits, activation='linear', name="Roll_outputb")(z1)
    z = layers.Dense(1, activation='linear', name="Roll_output")(z)
     
    losses = {
    	"Yaw_output": "mean_squared_error",
    	"Pitch_output":"mean_squared_error",
        "Roll_output": "mean_squared_error",
        "Yaw_outputb": tf.keras.losses.BinaryCrossentropy(from_logits=True),# "mean_squared_error",
    	"Pitch_outputb": tf.keras.losses.BinaryCrossentropy(from_logits=True),# "mean_squared_error",
        "Roll_outputb": tf.keras.losses.BinaryCrossentropy(from_logits=True)# "mean_squared_error"
    }
    lossWeights = {"Yaw_output": 0.0, "Pitch_output": 0.0, "Roll_output": 0.0,"Yaw_outputb": 1.0, "Pitch_outputb": 1.0, "Roll_outputb": 1.0}
    
    model = Model(inputs=model.input, outputs=[x,y,z,xb,yb,zb])
    model.compile(loss=losses, loss_weights=lossWeights, metrics=metrics, optimizer=optimizer)
    model.summary()
    model.load_weights(model_filename)
    val_dg = DirectoryDataGenerator(test_files, test_labels, test_Bbox, target_sizes=image_size, gamma=gamma,augmentor=False,
                                    preprocessors=res50_pp_input, batch_size=batch_size, shuffle=False, verbose=verbose,codefile=code_pkl,code_bits=code_bits)#format validation data
    acc = validateRegression(val_dg, model,code_tensor,code_name)
    print(acc)
    
    
    del model
    K.clear_session()

    
