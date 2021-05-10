# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import sys
from . import tf_util_tf2

def MLP (scope_name, features, layer_dims, bn_mode=False, train_mode=True,
         reg = tf.keras.regularizers.l2(1e-3), reuse_mode=False):
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        for i, num_outputs in enumerate(layer_dims[:-1]):
            features = tf_util_tf2.dense('dense'+str(i),features,num_outputs,train_mode,
                                         reuse_mode,tf.nn.relu,use_bn=bn_mode,regularizer=reg)
        outputs = tf_util_tf2.dense('dense'+str(i+1), features, layer_dims[-1],train_mode,
                                    reuse_mode,use_bn=bn_mode,regularizer=reg)
    return outputs

def sharedMLP_fullin (scope_name, inputs, layer_dims, bn_mode=False, train_mode=True,
                      reg = tf.keras.regularizers.l2(1e-3), reuse_mode=False):
    """
    inputs:[b,num_points,3]
    Features: [batch_size, 1, layer_dims[-1]], outputs: [b, num_points, layer_dims[-1]]
    """
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf_util_tf2.conv1d('sfc'+str(i), inputs, num_out_channel, train_mode, reuse_mode,
                                    tf.nn.relu, use_bn = bn_mode, regularizer=reg)
        outputs = tf_util_tf2.conv1d('sfc'+str(len(layer_dims)-1), inputs, layer_dims[-1], train_mode,
								 reuse_mode, use_bn=False, regularizer=reg)
        features = tf.reduce_max(outputs, axis=1, keepdims=True)
    return outputs,features

def sharedMLP (scope_name,inputs, npts, layer_dims, bn_mode=False, train_mode=True,
               reg = tf.keras.regularizers.l2(1e-3), reuse_mode=False):
    """
    inputs:[1,None,3],npts:[batch_size]
    Features: [batch_size, 1, layer_dims[-1]], outputs: [1, None, layer_dims[-1]]
    """
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf_util_tf2.conv1d('sfc'+str(i), inputs, num_out_channel,train_mode, reuse_mode,
                                        tf.nn.relu, use_bn = bn_mode,regularizer=reg)
        outputs = tf_util_tf2.conv1d('sfc'+str(len(layer_dims)-1), inputs, layer_dims[-1],train_mode,
                                     reuse_mode,use_bn=bn_mode,regularizer=reg)
        features = tf_util_tf2.point_maxpool(outputs, npts, keepdims=True)
    return outputs,features

def sharedMLP_simple (scope_name, inputs, layer_dims, bn_mode=False, train_mode=True,
                      reg = tf.keras.regularizers.l2(1e-3), reuse_mode=False):
    """
    inputs:[1,None,3],npts:[batch_size]
    Features: [batch_size, 1, layer_dims[-1]], outputs: [1, None, layer_dims[-1]]
    """
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf_util_tf2.conv1d('sfc'+str(i), inputs, num_out_channel,train_mode, reuse_mode,
                                        tf.nn.relu, use_bn = bn_mode,regularizer=reg)
        outputs = tf_util_tf2.conv1d('sfc'+str(len(layer_dims)-1), inputs, layer_dims[-1],train_mode,reuse_mode,
                                     use_bn=False,regularizer=reg)
    return outputs

def attnHead_all (hid, concat_fts, out_size, num_sample, activation, bn_mode, train_mode,
                  in_drop=0.0, coef_drop=0.0, reg = tf.keras.regularizers.l2(1e-3), reuse_mode=False):
    """
    concat_fts: #[batch_size, num_sample, 512+512] 
    hid: head_id
    Returns: #[b, num_sample, out_size]
    """
      
    name = 'attn'+str(hid)
    with tf.variable_scope(name, reuse=reuse_mode):
        if in_drop != 0.0:
            concat_fts = tf.layers.dropout(concat_fts, rate=in_drop, training=train_mode)
        
        #[batch_size, num_sample, out_size]
        seq_fts = sharedMLP_simple ('fts', concat_fts, [out_size,out_size], bn_mode, train_mode)
        
        f_1 = sharedMLP_simple ('f_1', seq_fts, [out_size/4,1], bn_mode, train_mode)   #[batch_size, num_sample, 1]
        f_2 = sharedMLP_simple ('f_2', seq_fts, [out_size/4,1], bn_mode, train_mode)   #[batch_size, num_sample, 1]
        logits = f_1 + tf.transpose(f_2, [0,2,1])   	    #[batch_size, num_sample, num_sample]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))     #[batch_size, num_sample, num_sample]
        
        if coef_drop != 0.0:
            coefs = tf.layers.dropout(coefs, rate=coef_drop, training=train_mode)
        if in_drop != 0.0:
            seq_fts = tf.layers.dropout(seq_fts, rate=in_drop, training=train_mode)
        
        vals = tf.matmul(coefs, seq_fts)  #[batch_size, num_sample, out_size]
        vals_wres = vals + sharedMLP_simple ('res', concat_fts, [out_size], bn_mode, train_mode)  # residual connect 
    
    return activation(vals_wres) #[batch_size, num_sample, out_size]
