# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:00:12 2019

@author: Sahil

NOTE: WORKING WITH TENSORFLOW (2.3.0, non-GPU) with KERAS (2.4.3). WORKS BUT NOT WITH CHANNELS_FIRST ORDER (c, h, w)
      WORKING WITH TENSORFLOW (2.4.0, GPU: 3080) with KERAS (2.4.3).  ONLY tried CHANNELS_LAST_ORDER, haven't tried first_order yet.


      WHEN USING TENSORFLOW-GPU (1.14.0), use KERAS (2.3.1) OR ELSE WILL GET COMPATIBILITY ERROR ("NEED TF >= 2.2")
      TENSORFLOW-GPU NOT WORKING DUE TO CUDNN (7...) FAILING ON CONV2D...
"""

import os
import sys
import semver
from tensorflow import __version__ as tf_ver
from take_snap_webcam import take_snapshot
from cost import compute_triplet_cost
from facenet_model import model
from img_utils import verify_name, build_database
from face_utils import pretrain_FaceNet
from verify import verify_image
from change_permissions import change_permissions, recursive_change_permissions


#check whether using tensorflow 1 or 2 and determine which functions to use 
tf_version = semver.VersionInfo.parse(tf_ver)
if tf_version.major == 1:
    import keras
else:
    import tensorflow.keras as keras


# MACROS
cwd = os.getcwd()
database_path = os.path.join(cwd, 'images/database')
weight_path = os.path.join(cwd, 'weights')
weights_filetype = '.csv'

image_path = os.path.join(cwd, 'images/camera/camera_0.jpg')
image_height = 96
image_width = 96
channel_order = 'channels_last'

distance_thresh = 0.63

margin = 0.2        #triplet loss margin hyperparameter

file_to_restrict = os.path.join(cwd, 'test.py')





# MAIN


#added this code for more recent versions of tensorflow (v2.4.0) and Keras (v2.4.3) and using RTX 3080 card
#when trying predict_on_batch, keep getting -> NotFoundError:  No algorithm worked!
#the following snippet of code fixes this issue and the code runs now

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



identity = input("\n\nPlease state your name: ")

#check if name is among database of authenticated users
if not verify_name(database_path, identity):
    change_permissions(file_to_restrict, 000)
    sys.exit(identity + "! You are not among the list of authenticated users!")

take_snapshot()

#reshape image depending on first/last channel selection
if channel_order == 'channels_last':
    img_shape = (image_height, image_width, 3)
elif channel_order == 'channels_first':
    img_shape = (3, image_height, image_width)

print("\nCurrently building the FaceNet model...\n")
FRmodel = model(img_shape, channel_order)

FRmodel.compile(optimizer = 'adam', loss = compute_triplet_cost, metrics = ['accuracy'])

#load parameters
print("\nCurrently loading pre-trained parameters to FaceNet...\n")
pretrain_FaceNet(FRmodel, weight_path, weights_filetype)

print("\nCurrently building the database of authenticated user encodings...\n")
database = build_database(database_path, image_height, image_width, channel_order, FRmodel)

print("\nVerifying Person...\n")
verified = verify_image(image_path, image_height, image_width, identity, database, channel_order, FRmodel, distance_thresh)

if verified == False:
    change_permissions(file_to_restrict, 000)
else:
    change_permissions(file_to_restrict, 0o754)
