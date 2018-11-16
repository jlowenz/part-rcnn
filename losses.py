"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import glob
import random
import math
import datetime
import time
import itertools
import json
import re
import pdb
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
import keras.regularizers as kr

from graph_util import *

from partnet.loss.pose_loss import camera_pose_loss
from partnet.config import Config

############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    # clean up bogus logits
    # how the F is it possible for there to be NaN after this?????
    #os = tf.ones_like(rpn_class_logits[:,0])        
    #rpn_class_logits_lo = tf.where(tf.is_nan(rpn_class_logits[:,0]), -os, rpn_class_logits[:,0])
    #rpn_class_logits_lo = tf.where(tf.is_finite(rpn_class_logits_lo), rpn_class_logits_lo, -os)
    #rpn_class_logits_hi = tf.where(tf.is_nan(rpn_class_logits[:,1]), os, rpn_class_logits[:,1])
    #rpn_class_logits_hi = tf.where(tf.is_finite(rpn_class_logits_hi), rpn_class_logits_hi, os)    
    #rpn_class_logits = tf.stack([rpn_class_logits_lo, rpn_class_logits_hi], axis=1)
    # select valid anchors
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Crossentropy loss
    #anchor_class = tf.Print(anchor_class, [anchor_class], "anchor class: ", summarize=2000)
    #rpn_class_logits = tf.Print(rpn_class_logits, [rpn_class_logits], "rpn class logits: ", summarize=2000)
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)    
    #loss = tf.Print(loss, [loss], "Loss: ", summarize=2000)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0)) + K.sum(reg_losses)
    #loss = tf.Print(loss, [loss], "Loss: ", summarize=2000)
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    # TODO: use smooth_l1_loss() rather than reimplementing here
    #       to reduce code duplication
    diff = K.abs(target_bbox - rpn_bbox)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    target_class_ids = tf.cast(target_class_ids, 'int64')
    #active_class_ids = tf.Print(active_class_ids, [active_class_ids],
    #"active class ids: ", summarize=100)

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    batch_active = tf.reduce_max(active_class_ids, axis=0)
    #print("batch_active {}".format(batch_active.shape))
    pred_active = tf.gather(batch_active, pred_class_ids)
    #print("pred_active {}".format(pred_active.shape))

    #target_class_ids = tf.Print(target_class_ids, [target_class_ids],
    #                            "Target class ids: ", summarize=100)
    #pred_class_logits = tf.Print(pred_class_logits, [pred_class_logits],
    #                             "Pred class logits: ", summarize=100)
    #pred_active = tf.Print(pred_active, [pred_active],
    #                       "Pred active: ", summarize=100)
    
    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.    
    #loss = tf.Print(loss, [loss], "\nLOSS: ", summarize=14)
    #pred_active = tf.Print(pred_active, [pred_active], "\nPred active: ", summarize=14)
    loss = tf.reduce_sum(loss) / (tf.reduce_sum(pred_active) + 1e-6)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss

def object_pose_loss(gt_pose, pred_trans, pred_quat):
    print("gt_pose shape: {}".format(gt_pose.shape))
    gt_trans = gt_pose[:,4:]
    gt_quat = gt_pose[:,:4]
    # reuse our other code
    return camera_pose_loss([pred_trans, pred_quat, gt_trans, gt_quat])

def compute_prim_targets(gt_prims, target_class_ids):
    cfg = Config()
    # we need to select the gt_prims that correspond to the class_ids for each
    # ROI
    #
    # gt_prims: [num_classes, num_params]
    #    num_classes does not include the unlabeled class here!
    #    num_params *includes* the inital presence indicator
    # target_masks: [batch, num_rois, h, w]
    # target_class_ids: [batch, num_rois]
    tshape = tf.shape(target_class_ids)
    print("tshape: {}".format(tshape))
    target_class_ids = tf.cast(tf.reshape(target_class_ids, (-1,)), tf.int32)
    

    # fix the gt_prims for our need
    B = tf.shape(gt_prims)[0]
    P = gt_prims.shape[1]
    Q = gt_prims.shape[2]
    # add a row to represent the unlabeled primitive
    gt_prims = tf.concat([tf.zeros([B,1,Q]),gt_prims], axis=1)
    # remove the first column representing the presence indicator
    gt_prims = gt_prims[:,:,1:]
    
    # what we need is a [batch, num_rois, num_params]
    # array
    print("target_class_ids {}".format(target_class_ids.shape))
    target_prims = tf.gather(gt_prims, target_class_ids)
    print("target_prims shape {}".format(target_prims.shape))
    target_prims = tf.reshape(target_prims, (tshape[0],tshape[1],Q-1))
    return target_prims    

# target_prims must be computed based on the PRED_PRIMS
# i.e. for each pred_prim, we must 
# first part primitive loss
def primitive_direct_loss(target_prims, target_class_ids, pred_prims):
    '''
    target_prims: [batch, num_rois, num_params]
    target_class_ids: [batch, num_rois]
        integer class ids. 
    pred_prims: [batch, num_rois, num_classes, num_params]

    How the heck will this work?
    '''
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    target_shape = target_prims.shape
    target_prims = tf.reshape(target_prims, [-1, target_shape[2]])
    # target_prims is [BN, Q]
    
    pred_shape = pred_prims.shape
    print("pred shape: {}".format(pred_shape))
    pred_prims = tf.reshape(pred_prims, [-1, pred_shape[2], pred_shape[3]])
    print("pred_prims: {}".format(pred_prims.shape))
    # now, pred_prims is [BN, P, Q]
    
    y_true = tf.gather(target_prims, positive_ix)
    y_pred = tf.gather_nd(pred_prims, indices)
    print("y_true: {}".format(y_true.shape))
    print("y_pred: {}".format(y_pred.shape))

    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return loss

    
