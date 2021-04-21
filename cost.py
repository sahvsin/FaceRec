# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:25:11 2019

@author: Sahil
"""

import tensorflow as tf




def compute_triplet_cost(true_triplet, pred_triplet, margin = 0.2):
    
    """
    computes the cost function on a triplet of encoded images (anchor, positive, and negative) by computing the 
    difference between the (squared-norm) distances between the positive and anchor and between the negative and 
    anchor plus a manually chosen margin such that this expression evaluates to less-than or equal-to 0
    (derived from rearringing the original inequality where the distance between positive and anchor must be less than
    or eqaul to the distance between negative and anchor minus the margin)
    
    Inputs:
    true_triplet -- the true labels for the anchor, positive, and negative images 
                    anchor: reference image
                    positive: image of same person/thing as anchor (should have small distance)
                    negative: image of different person/thing as anchor (should have large distance)
    pred_triplet -- list of the encdoings/names of the chosen content layer 
    margin -- hyperparameter that modifies cost to make sure the distances are different (by at least said margin)
    """
    
    # distill the anchor, positive, and negative image encodings
    anchor, positive, negative = pred_triplet[0], pred_triplet[1], pred_triplet[2]
    
    #compute the (encoding) distance between the (encoded) anchor and positive images
    pos_dist = tf.reduce_sum(tf.compat.v1.squared_difference(anchor, positive), axis = -1)
    
    #compute the (encoding) distance between the (encoded) anchor and negative images
    neg_dist = tf.reduce_sum(tf.compat.v1.squared_difference(anchor, negative), axis = -1)
    
    #compute the sum of the difference between the two distances and margin
    cost = (pos_dist - neg_dist) + margin
    
    #compute the sum of the maximum between 0 and the computed cost of all examples (treat negative costs as 0)
    triplet_cost = tf.reduce_sum(tf.maximum(cost, 0))
    
    return triplet_cost
