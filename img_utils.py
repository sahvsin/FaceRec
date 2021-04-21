# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:37:50 2019

@author: Sahil
"""


import os
import semver
import numpy as np
import cv2
from tensorflow import __version__ as tf_ver


#check whether using tensorflow 1 or 2 and determine which functions to use 
tf_version = semver.VersionInfo.parse(tf_ver)
if tf_version.major == 1:
    import keras
else:
    import tensorflow.keras as keras



def int2float(x, n=12):
    
    '''
    converts array of uint8 (256 value) data to float between 0 and 1 rounded to the nth decimal
    
    Inputs:
    x -- the data (assumed uint8) to be converted to float between 0 and 1
    n -- number of decimals to round to

    Output:
    Float casting from a uint8
    '''
    
    return np.around(x/255.0, decimals=n)


def mod_img(img, new_h, new_w, dtype = 'int'):

    '''
    modifies the provided to image to a new image with the provided dimensions and casts it to the provided datatype
    assumes the use of 'cv2.imread' to originally load the image (defaults BGR) so rearranges to RGB

    Inputs:
    img -- the image to modify
    new_h -- the new height 
    new_w -- the new width
    dtype -- the new datatype (default is 'int')

    Outputs:
    the resized, RGB, float/int image
    '''

    img = cv2.resize(img, (new_w, new_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dtype == 'float':
        img = int2float(img, 12)
    return img


def encode_img(img_path, new_h, new_w, channel_order, model):
    
    '''
    loads an image, modifies it given modification parameters, and encodes it

    Inputs:
    img_path -- path to the image file
    new_h -- modify the image to this height
    new_w -- modify the image to this width
    channel_order -- order of color channels to process the image (RGB vs BGR)
    model -- the face recongition model

    Output: 
    A 128-point vector encoding of the resized input image
    '''

    orig_img  = cv2.imread(img_path) 
    img = mod_img(orig_img, new_h, new_w, dtype = 'float')
    train_input = np.array([img])
    if channel_order == 'channels_first':
        train_input = np.transpose(train_input, (0, 3, 1, 2))
    encoding = model.predict_on_batch(train_input)
    return encoding
    

    
def verify_name(path, name_to_verify):
    
    '''
    checks if the input identity is among the verified users listed in the database

    Inputs:
    path -- path to the database of images of verified users
    name_to_verify -- the username input by the current user to verify

    Output:
    Boolean determining whether the username is among the list of verified users
    '''

    #go through every file in the database directory and check if the username matches with the inputted name
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            #filename = filename.rstrip('\n\r')
            if name_to_verify == filename.split('.')[0]:
                return True
    return False
    




def build_database(path, new_h, new_w, channel_order, model):

    '''
    Builds a Python dictionary that maps each verified user (name) to the encoding of their respective image using the face recognition model
    
    Inputs:
    path -- path of the database of images of verified users
    new_h -- height of the image after resizing
    new_w -- width of the image after resizing
    channel_order -- order of color channels to process the image (RGB vs BGR)
    model -- the face recogniton model

    Output:
    Hashmap (Python dictionary) mapping each verified username to the encoding of their image
    '''
    
    database = dict()
    
    #go through every file in the database directory and process each image
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            #filename = filename.rstrip('\n\r')
            label = filename.split('.')[0]
            filepath = os.path.join(path, filename)
            
            #save the encodings
            database[label] = encode_img(filepath, new_h, new_w, channel_order, model)
            
            
    #print(database)
    return database








