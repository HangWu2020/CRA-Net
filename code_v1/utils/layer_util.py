#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:42:19 2020

@author: wuhang
"""

import tf_util
import tensorflow as tf

def MLP (scope_name, features, layer_dims, bn_mode=False, train_mode=True,
         reg = tf.contrib.layers.l2_regularizer(1e-3), reuse_mode=False):
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        for i, num_outputs in enumerate(layer_dims[:-1]):
            features = tf_util.dense('dense'+str(i),features,num_outputs,train_mode,
                                     reuse_mode,tf.nn.relu,use_bn=bn_mode,regularizer=reg)
        outputs = tf_util.dense('dense'+str(i+1), features, layer_dims[-1],train_mode,
                                reuse_mode,use_bn=bn_mode,regularizer=reg)
    return outputs

def sharedMLP (scope_name,inputs, npts, layer_dims, bn_mode=False, train_mode=True,
               reg = tf.contrib.layers.l2_regularizer(1e-3), reuse_mode=False):
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf_util.conv1d('sfc'+str(i), inputs, num_out_channel,train_mode, reuse_mode,
                                    tf.nn.relu, use_bn = bn_mode,regularizer=reg)
        outputs = tf_util.conv1d('sfc'+str(len(layer_dims)-1), inputs, layer_dims[-1],train_mode,
                                 reuse_mode,use_bn=bn_mode,regularizer=reg)
        features = tf_util.point_maxpool(outputs, npts, keepdims=True)
    return outputs,features

def sharedMLP_simple (scope_name, inputs, layer_dims, bn_mode=False, train_mode=True,
                      reg = tf.contrib.layers.l2_regularizer(1e-3), reuse_mode=False):
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf_util.conv1d('sfc'+str(i), inputs, num_out_channel,train_mode, reuse_mode,
                                    tf.nn.relu, use_bn = bn_mode,regularizer=reg)
        outputs = tf_util.conv1d('sfc'+str(len(layer_dims)-1), inputs, layer_dims[-1],train_mode,reuse_mode,
                                 use_bn=False,regularizer=reg)
    return outputs

def attnHead_all (hid, concat_fts, out_size, num_sample, activation, bn_mode, train_mode,
                  in_drop=0.0, coef_drop=0.0, reg = tf.contrib.layers.l2_regularizer(1e-3), reuse_mode=False):
      
    name = 'attn'+str(hid)
    with tf.variable_scope(name, reuse=reuse_mode):
        if in_drop != 0.0:
            concat_fts = tf.layers.dropout(concat_fts, rate=in_drop, training=train_mode)
        
        seq_fts = sharedMLP_simple ('fts', concat_fts, [out_size,out_size], bn_mode, train_mode)
        
        f_1 = sharedMLP_simple ('f_1', seq_fts, [out_size/4,1], bn_mode, train_mode)
        f_2 = sharedMLP_simple ('f_2', seq_fts, [out_size/4,1], bn_mode, train_mode)
        logits = f_1 + tf.transpose(f_2, [0,2,1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
        
        if coef_drop != 0.0:
            coefs = tf.layers.dropout(coefs, rate=coef_drop, training=train_mode)
        if in_drop != 0.0:
            seq_fts = tf.layers.dropout(seq_fts, rate=in_drop, training=train_mode)
        
        vals = tf.matmul(coefs, seq_fts)
        vals_wres = vals + sharedMLP_simple ('res', concat_fts, [out_size], bn_mode, train_mode)
    
    return activation(vals_wres)
