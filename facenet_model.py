# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:32:06 2019

@author: Sahil
"""


import semver
import numpy as np
from tensorflow import __version__ as tf_ver

#check whether using tensorflow 1 or 2 and determine which functions to use 
tf_version = semver.VersionInfo.parse(tf_ver)
if tf_version.major == 1:
    import keras
    from keras.layers.normalization import BatchNormalization
else:
    import tensorflow.keras as keras
    from keras.layers import BatchNormalization
#also change BatchNormalization calls down below in the "model" function
#older keras (version < 2) -> keras.layers.normalization.BatchNormalization 
#newer keras or tf.keras -> tf.keras.layers.BatchNormalization


#labels/keys I use for accessing and assigning weights
Weights = [
        'conv1', 'bn1', 
        'conv2', 'bn2',
        'conv3', 'bn3',
        'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
  	'inception_3a_pool_conv', 'inception_3a_pool_bn',
  	'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
  	'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
  	'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
  	'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
  	'inception_3b_pool_conv', 'inception_3b_pool_bn',
  	'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
  	'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
  	'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
  	'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
  	'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
  	'inception_4a_pool_conv', 'inception_4a_pool_bn',
  	'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
  	'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
  	'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
  	'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
  	'inception_5a_pool_conv', 'inception_5a_pool_bn',
  	'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
  	'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
  	'inception_5b_pool_conv', 'inception_5b_pool_bn',
  	'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
        'dense'
        ]



#hashmap of each layer (name) and their shape/size
conv_shape = {
        'conv1': [64, 3, 7, 7],
  	'conv2': [64, 64, 1, 1],
  	'conv3': [192, 64, 3, 3],
  	'inception_3a_1x1_conv': [64, 192, 1, 1],
  	'inception_3a_pool_conv': [32, 192, 1, 1],
  	'inception_3a_5x5_conv1': [16, 192, 1, 1],
  	'inception_3a_5x5_conv2': [32, 16, 5, 5],
  	'inception_3a_3x3_conv1': [96, 192, 1, 1],
  	'inception_3a_3x3_conv2': [128, 96, 3, 3],
  	'inception_3b_3x3_conv1': [96, 256, 1, 1],
  	'inception_3b_3x3_conv2': [128, 96, 3, 3],
  	'inception_3b_5x5_conv1': [32, 256, 1, 1],
  	'inception_3b_5x5_conv2': [64, 32, 5, 5],
  	'inception_3b_pool_conv': [64, 256, 1, 1],
  	'inception_3b_1x1_conv': [64, 256, 1, 1],
  	'inception_3c_3x3_conv1': [128, 320, 1, 1],
  	'inception_3c_3x3_conv2': [256, 128, 3, 3],
  	'inception_3c_5x5_conv1': [32, 320, 1, 1],
  	'inception_3c_5x5_conv2': [64, 32, 5, 5],
  	'inception_4a_3x3_conv1': [96, 640, 1, 1],
  	'inception_4a_3x3_conv2': [192, 96, 3, 3],
  	'inception_4a_5x5_conv1': [32, 640, 1, 1,],
  	'inception_4a_5x5_conv2': [64, 32, 5, 5],
  	'inception_4a_pool_conv': [128, 640, 1, 1],
  	'inception_4a_1x1_conv': [256, 640, 1, 1],
  	'inception_4e_3x3_conv1': [160, 640, 1, 1],
  	'inception_4e_3x3_conv2': [256, 160, 3, 3],
  	'inception_4e_5x5_conv1': [64, 640, 1, 1],
  	'inception_4e_5x5_conv2': [128, 64, 5, 5],
  	'inception_5a_3x3_conv1': [96, 1024, 1, 1],
  	'inception_5a_3x3_conv2': [384, 96, 3, 3],
  	'inception_5a_pool_conv': [96, 1024, 1, 1],
  	'inception_5a_1x1_conv': [256, 1024, 1, 1],
  	'inception_5b_3x3_conv1': [96, 736, 1, 1],
  	'inception_5b_3x3_conv2': [384, 96, 3, 3],
  	'inception_5b_pool_conv': [96, 736, 1, 1],
  	'inception_5b_1x1_conv': [256, 736, 1, 1],
        }




def inception_block_3a(X, channel_order = 'channels_last', bn_axis = 3):
    
    '''
    first part of the first inception block

    Inputs
    X -- tensor being processed
    channel_order -- whether number of channels are first or last (c,l,w or l,w,c)
    bn_axis -- batch normalization axis which should be the features axis (axis with number of channels, so 1 or 3 depending on channel order)

    Output
    result from the concatenation of the branched convolutions in this inception    
    '''

    X_3x3 = keras.layers.Conv2D(96, (1, 1), data_format = channel_order, name = 'inception_3a_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3a_3x3_bn1')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    X_3x3 = keras.layers.ZeroPadding2D(padding = (1, 1), data_format = channel_order)(X_3x3)
    X_3x3 = keras.layers.Conv2D(128, (3, 3), data_format = channel_order, name = 'inception_3a_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3a_3x3_bn2')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    
    X_5x5 = keras.layers.Conv2D(16, (1, 1), data_format = channel_order, name = 'inception_3a_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3a_5x5_bn1')(X_5x5)
    X_5x5 = keras.layers.Activation('relu')(X_5x5)
    X_5x5 = keras.layers.ZeroPadding2D(padding = (2, 2), data_format = channel_order)(X_5x5)
    X_5x5 = keras.layers.Conv2D(32, (5, 5), data_format = channel_order, name = 'inception_3a_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3a_5x5_bn2')(X_5x5)
    X_5x5 = keras.layers.Activation('relu')(X_5x5)
    
    X_pool = keras.layers.MaxPooling2D(pool_size = 3, strides = 2, data_format = channel_order)(X)
    X_pool = keras.layers.Conv2D(32, (1, 1), data_format = channel_order, name = 'inception_3a_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3a_pool_bn')(X_pool)
    X_pool = keras.layers.Activation('relu')(X_pool)
    X_pool = keras.layers.ZeroPadding2D(padding = ((3, 4), (3, 4)), data_format = channel_order)(X_pool)
    
    X_1x1 = keras.layers.Conv2D(64, (1, 1), data_format = channel_order, name = 'inception_3a_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3a_1x1_bn')(X_1x1)
    X_1x1 = keras.layers.Activation('relu')(X_1x1)
    
    inception_3a = keras.layers.concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis = bn_axis)
    
    return inception_3a



def inception_block_3b(X, channel_order = 'channels_last', bn_axis = 3):
    
    '''
    second part of the first inception block

    Inputs
    X -- tensor being processed
    channel_order -- whether number of channels are first or last (c,l,w or l,w,c)
    bn_axis -- batch normalization axis which should be the features axis (axis with number of channels, so 1 or 3 depending on channel order)

    Output
    result from the concatenation of the branched convolutions in this inception 
    '''
    
    X_3x3 = keras.layers.Conv2D(96, (1, 1), data_format = channel_order, name = 'inception_3b_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3b_3x3_bn1')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    X_3x3 = keras.layers.ZeroPadding2D(padding = (1, 1), data_format = channel_order)(X_3x3)
    X_3x3 = keras.layers.Conv2D(128, (3, 3), data_format = channel_order, name = 'inception_3b_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3b_3x3_bn2')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    
    X_5x5 = keras.layers.Conv2D(32, (1, 1), data_format = channel_order, name = 'inception_3b_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3b_5x5_bn1')(X_5x5)
    X_5x5 = keras.layers.Activation('relu')(X_5x5)
    X_5x5 = keras.layers.ZeroPadding2D(padding = (2, 2), data_format = channel_order)(X_5x5)
    X_5x5 = keras.layers.Conv2D(64, (5, 5), data_format = channel_order, name = 'inception_3b_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3b_5x5_bn2')(X_5x5)
    X_5x5 = keras.layers.Activation('relu')(X_5x5)
    
    X_pool = keras.layers.AveragePooling2D(pool_size = (3, 3), strides = (3, 3), data_format = channel_order)(X)
    X_pool = keras.layers.Conv2D(64, (1, 1), data_format = channel_order, name = 'inception_3b_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3b_pool_bn')(X_pool)
    X_pool = keras.layers.Activation('relu')(X_pool)
    X_pool = keras.layers.ZeroPadding2D(padding = (4, 4), data_format = channel_order)(X_pool)
    
    X_1x1 = keras.layers.Conv2D(64, (1, 1), data_format = channel_order, name = 'inception_3b_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3b_1x1_bn')(X_1x1)
    X_1x1 = keras.layers.Activation('relu')(X_1x1)
    
    inception_3b = keras.layers.concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis = bn_axis)
    
    return inception_3b



def inception_block_3c(X, channel_order = 'channels_last', bn_axis = 3):
    
    '''
    third part of the first inception block

    Inputs
    X -- tensor being processed
    channel_order -- whether number of channels are first or last (c,l,w or l,w,c)
    bn_axis -- batch normalization axis which should be the features axis (axis with number of channels, so 1 or 3 depending on channel order)

    Output
    result from the concatenation of the branched convolutions in this inception 
    '''
    
    X_3x3 = keras.layers.Conv2D(128, (1, 1), data_format = channel_order, name = 'inception_3c_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3c_3x3_bn1')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    X_3x3 = keras.layers.ZeroPadding2D(padding = (1, 1), data_format = channel_order)(X_3x3)
    X_3x3 = keras.layers.Conv2D(256, (3, 3), strides = (2, 2), data_format = channel_order, name = 'inception_3c_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3c_3x3_bn2')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    
    X_5x5 = keras.layers.Conv2D(32, (1, 1), data_format = channel_order, name = 'inception_3c_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3c_5x5_bn1')(X_5x5)
    X_5x5 = keras.layers.Activation('relu')(X_5x5)
    X_5x5 = keras.layers.ZeroPadding2D(padding = (2, 2), data_format = channel_order)(X_5x5)
    X_5x5 = keras.layers.Conv2D(64, (5, 5), strides = (2, 2), data_format = channel_order, name = 'inception_3c_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_3c_5x5_bn2')(X_5x5)
    X_5x5 = keras.layers.Activation('relu')(X_5x5)
    
    X_pool = keras.layers.MaxPooling2D(pool_size = 3, strides = 2, data_format = channel_order)(X)
    X_pool = keras.layers.ZeroPadding2D(padding = ((0, 1), (0, 1)), data_format = channel_order)(X_pool)
    
    inception_3c = keras.layers.concatenate([X_3x3, X_5x5, X_pool], axis = bn_axis)
    
    return inception_3c




def inception_block_4a(X, channel_order = 'channels_last', bn_axis = 3):
    
    '''
    first part of the second inception block

    Inputs
    X -- tensor being processed
    channel_order -- whether number of channels are first or last (c,l,w or l,w,c)
    bn_axis -- batch normalization axis which should be the features axis (axis with number of channels, so 1 or 3 depending on channel order)

    Output
    result from the concatenation of the branched convolutions in this inception 
    '''
    
    X_3x3 = keras.layers.Conv2D(96, (1, 1), data_format = channel_order, name = 'inception_4a_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_4a_3x3_bn1')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    X_3x3 = keras.layers.ZeroPadding2D(padding = (1, 1), data_format = channel_order)(X_3x3)
    X_3x3 = keras.layers.Conv2D(192, (3, 3), data_format = channel_order, name = 'inception_4a_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_4a_3x3_bn2')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    
    X_5x5 = keras.layers.Conv2D(32, (1, 1), data_format = channel_order, name = 'inception_4a_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_4a_5x5_bn1')(X_5x5)
    X_5x5 = keras.layers.Activation('relu')(X_5x5)
    X_5x5 = keras.layers.ZeroPadding2D(padding = (2, 2), data_format = channel_order)(X_5x5)
    X_5x5 = keras.layers.Conv2D(64, (5, 5), data_format = channel_order, name = 'inception_4a_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_4a_5x5_bn2')(X_5x5)
    X_5x5 = keras.layers.Activation('relu')(X_5x5)
    
    X_pool = keras.layers.AveragePooling2D(pool_size = 3, strides = 3, data_format = channel_order)(X)
    X_pool = keras.layers.Conv2D(128, (1, 1), data_format = channel_order, name = 'inception_4a_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_4a_pool_bn')(X_pool)
    X_pool = keras.layers.Activation('relu')(X_pool)
    X_pool = keras.layers.ZeroPadding2D((2, 2), data_format = channel_order)(X_pool)
    
    X_1x1 = keras.layers.Conv2D(256, (1, 1), data_format = channel_order, name = 'inception_4a_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_4a_1x1_bn')(X_1x1)
    X_1x1 = keras.layers.Activation('relu')(X_1x1)
    
    inception_4a = keras.layers.concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis = bn_axis)
    
    return inception_4a




def inception_block_4e(X, channel_order = 'channels_last', bn_axis = 3):
    
    '''
    second part of the second inception block

    Inputs
    X -- tensor being processed
    channel_order -- whether number of channels are first or last (c,l,w or l,w,c)
    bn_axis -- batch normalization axis which should be the features axis (axis with number of channels, so 1 or 3 depending on channel order)

    Output
    result from the concatenation of the branched convolutions in this inception 
    '''
    
    X_3x3 = keras.layers.Conv2D(160, (1, 1), data_format = channel_order, name = 'inception_4e_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_4e_3x3_bn1')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    X_3x3 = keras.layers.ZeroPadding2D(padding = (1, 1), data_format = channel_order)(X_3x3)
    X_3x3 = keras.layers.Conv2D(256, (3, 3), strides = (2, 2), data_format = channel_order, name = 'inception_4e_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_4e_3x3_bn2')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    
    X_5x5 = keras.layers.Conv2D(64, (1, 1), data_format = channel_order, name = 'inception_4e_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_4e_5x5_bn1')(X_5x5)
    X_5x5 = keras.layers.Activation('relu')(X_5x5)
    X_5x5 = keras.layers.ZeroPadding2D(padding = (2, 2), data_format = channel_order)(X_5x5)
    X_5x5 = keras.layers.Conv2D(128, (5, 5), strides = (2, 2), data_format = channel_order, name = 'inception_4e_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_4e_5x5_bn2')(X_5x5)
    X_5x5 = keras.layers.Activation('relu')(X_5x5)
    
    X_pool = keras.layers.MaxPooling2D(pool_size = 3, strides = 2, data_format = channel_order)(X)
    X_pool = keras.layers.ZeroPadding2D(padding = ((0, 1), (0, 1)), data_format = channel_order)(X_pool)
    
    inception_4e = keras.layers.concatenate([X_3x3, X_5x5, X_pool], axis = bn_axis)
    
    return inception_4e




def inception_block_5a(X, channel_order = 'channels_last', bn_axis = 3):
    
    '''
    first part of the third inception block

    Inputs
    X -- tensor being processed
    channel_order -- whether number of channels are first or last (c,l,w or l,w,c)
    bn_axis -- batch normalization axis which should be the features axis (axis with number of channels, so 1 or 3 depending on channel order)

    Output
    result from the concatenation of the branched convolutions in this inception 
    '''
    
    X_3x3 = keras.layers.Conv2D(96, (1, 1), data_format = channel_order, name = 'inception_5a_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_5a_3x3_bn1')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    X_3x3 = keras.layers.ZeroPadding2D(padding = (1, 1), data_format = channel_order)(X_3x3)
    X_3x3 = keras.layers.Conv2D(384, (3, 3), data_format = channel_order, name = 'inception_5a_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_5a_3x3_bn2')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    
    X_pool = keras.layers.AveragePooling2D(pool_size = 3, strides = 3, data_format = channel_order)(X)
    X_pool = keras.layers.Conv2D(96, (1, 1), data_format = channel_order, name = 'inception_5a_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_5a_pool_bn')(X_pool)
    X_pool = keras.layers.Activation('relu')(X_pool)
    X_pool = keras.layers.ZeroPadding2D((1, 1), data_format = channel_order)(X_pool)
    
    X_1x1 = keras.layers.Conv2D(256, (1, 1), data_format = channel_order, name = 'inception_5a_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_5a_1x1_bn')(X_1x1)
    X_1x1 = keras.layers.Activation('relu')(X_1x1)
    
    inception_5a = keras.layers.concatenate([X_3x3, X_pool, X_1x1], axis = bn_axis)
    
    return inception_5a




def inception_block_5b(X, channel_order = 'channels_last', bn_axis = 3):
    
    '''
    second part of the third inception block

    Inputs
    X -- tensor being processed
    channel_order -- whether number of channels are first or last (c,l,w or l,w,c)
    bn_axis -- batch normalization axis which should be the features axis (axis with number of channels, so 1 or 3 depending on channel order)

    Output
    result from the concatenation of the branched convolutions in this inception 
    '''
    
    X_3x3 = keras.layers.Conv2D(96, (1, 1), data_format = channel_order, name = 'inception_5b_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_5b_3x3_bn1')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    X_3x3 = keras.layers.ZeroPadding2D(padding = (1, 1), data_format = channel_order)(X_3x3)
    X_3x3 = keras.layers.Conv2D(384, (3, 3), data_format = channel_order, name = 'inception_5b_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_5b_3x3_bn2')(X_3x3)
    X_3x3 = keras.layers.Activation('relu')(X_3x3)
    
    X_pool = keras.layers.MaxPooling2D(pool_size = 3, strides = 2, data_format = channel_order)(X)
    X_pool = keras.layers.Conv2D(96, (1, 1), data_format = channel_order, name = 'inception_5b_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_5b_pool_bn')(X_pool)
    X_pool = keras.layers.Activation('relu')(X_pool)
    X_pool = keras.layers.ZeroPadding2D((1, 1), data_format = channel_order)(X_pool)
    
    X_1x1 = keras.layers.Conv2D(256, (1, 1), data_format = channel_order, name = 'inception_5b_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'inception_5b_1x1_bn')(X_1x1)
    X_1x1 = keras.layers.Activation('relu')(X_1x1)
    
    inception_5b = keras.layers.concatenate([X_3x3, X_pool, X_1x1], axis = bn_axis)
    
    return inception_5b





def model(input_shape, channel_order = 'channels_last'):
    
    '''
    FaceNet keras model derived from Schroff's (et al) "FaceNet" paper

    Inputs:
    input_shape -- shape of the image(s) to process
    channel_order -- image dimensions order, last - (w, h, n_c), first - (n_c, w, h)

    Outputs:
    model -- keras model object instantiated with the input tensor and the FaceNet output specified by the sequence of layers
    '''

    #batch normailization axis should be the axis of features, which in this case is the axis with the number of channels
    if channel_order == 'channels_last':
        bn_axis = 3
    elif channel_order == 'channels_first':
        bn_axis = 1
    
    #create the input (tensor) to the network with the provided shape
    X_input = keras.layers.Input(input_shape)

    #Zero-Padding
    X = keras.layers.ZeroPadding2D((3, 3), data_format = channel_order)(X_input)
   
    #First-Block
    X = keras.layers.Conv2D(64, (7, 7), strides = (2, 2), data_format = channel_order, name = 'conv1')(X)
    X = BatchNormalization(axis = bn_axis, name = 'bn1')(X)
    X = keras.layers.Activation('relu')(X)
    
    #Zero-Padding and (Max) Pooling
    X = keras.layers.ZeroPadding2D((1, 1), data_format = channel_order)(X)
    X = keras.layers.MaxPooling2D((3, 3), strides = 2, data_format = channel_order)(X)
    
    #Second Block
    X = keras.layers.Conv2D(64, (1, 1), strides = (1, 1), data_format = channel_order, name = 'conv2')(X)
    X = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'bn2')(X)
    X = keras.layers.Activation('relu')(X)
    
    #Zero-Padding
    X = keras.layers.ZeroPadding2D((1, 1), data_format = channel_order)(X)
    
    #Third Block
    X = keras.layers.Conv2D(192, (3, 3), strides = (1, 1), data_format = channel_order, name = 'conv3')(X)
    X = BatchNormalization(axis = bn_axis, epsilon = .00001, name = 'bn3')(X)
    X = keras.layers.Activation('relu')(X)
    
    #Zero-Padding and (Max) Pooling
    X = keras.layers.ZeroPadding2D((1, 1), data_format = channel_order)(X)
    X = keras.layers.MaxPooling2D(pool_size = 3, strides = 2, data_format = channel_order)(X)
    
    #Third Block (Cont'd): First Inception Block
    X = inception_block_3a(X, channel_order, bn_axis)
    X = inception_block_3b(X, channel_order, bn_axis)
    X = inception_block_3c(X, channel_order, bn_axis)
    
    #Fourth Block: Second Inception Block
    X = inception_block_4a(X, channel_order, bn_axis)
    X = inception_block_4e(X, channel_order, bn_axis)
    
    #Fifth Block: Third Inception Block
    X = inception_block_5a(X, channel_order, bn_axis)
    X = inception_block_5b(X, channel_order, bn_axis)
    
    #TOP/FC LAYER
    X = keras.layers.AveragePooling2D(pool_size = (3, 3), strides = (1, 1), data_format = channel_order)(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(128, name = 'dense')(X)
    
    #create a Lambda/Functional layer that computes the L2 normalization of the output of the TOP/FC Layer (answer)
    X = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis = 1))(X)
    
    #instantiate a keras model object given the input and FaceNet output (specified by the sequence of layers above)
    model = keras.models.Model(inputs = X_input, outputs = X, name = 'FaceNet_Model')
    
    
    return model
    
