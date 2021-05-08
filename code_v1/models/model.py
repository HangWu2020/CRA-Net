#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:30:00 2020

@author: wuhang
"""

import sys
sys.path.append("../utils")
import tensorflow as tf
import tf_util
import layer_util
import numpy as np

class RAN_1l ():
    def __init__(self, inputs, npts, cores, nheads, alpha,
                 ball_index_0,ball_size_0, gt, gt_fine, bn, train):
        self.grid_size = 4
        self.grid_scale = 0.1
        self.batch_size = np.shape(npts)[0]
        self.num_coarse = 1024
        self.num_fine = self.num_coarse * (self.grid_size**2)
        self.global_codes = self.global_encoder(inputs, npts)
        self.local_codes = self.local_encoder(inputs,npts,cores,ball_index_0,ball_size_0)
        self.coarse,self.fine = self.gcn_decoder(self.global_codes, self.local_codes, cores, nheads, bn, train)
        self.loss_coarse, self.loss_fine, self.loss = self.get_loss(self.coarse, self.fine, gt, gt_fine, alpha)
                    
    def global_encoder(self, inputs, npts): 
        with tf.variable_scope ('global_encoder', reuse=False):
            raw_fts,max_fts = layer_util.sharedMLP ('ec_00',inputs, npts, [64,128])
            unpool_fts = tf_util.point_unpool(max_fts, npts) 
            concat_fts = tf.concat([raw_fts, unpool_fts], axis=2)
            _,global_fts = layer_util.sharedMLP ('ec_01',concat_fts, npts, [256,512])
        return global_fts
    
    def local_encoder (self, inputs, npts, cores,ball_index_0,ball_size_0):
        with tf.variable_scope ('local_encoder', reuse=False):
            cloud_exp_0 = tf.gather(inputs,ball_index_0,axis=1)
            raw_fts_0,max_fts_0 = layer_util.sharedMLP ('ec_10',cloud_exp_0, ball_size_0,[64,128])
            unpool_fts = tf_util.point_unpool(max_fts_0, ball_size_0)
            concat_fts = tf.concat([raw_fts_0, unpool_fts], axis=2)
            _,concat_fts = layer_util.sharedMLP ('ec_11',concat_fts, ball_size_0, [256,512])
            local_fts = tf.reshape(concat_fts,[self.batch_size,cores,512])
        return local_fts
    
    def gcn_decoder(self, glb_fts, loc_fts, cores, nheads, bn, train):
        head_pts = self.num_coarse/cores/nheads
        points_set = []
        glb_fts_exp = tf.tile(glb_fts,[1,cores,1])
        concat_fts = tf.concat([glb_fts_exp, loc_fts],axis=2)
        
        with tf.variable_scope ('decoder_coarse', reuse=False):
            for i in range (nheads):
                points_set.append (layer_util.attnHead_all (str(i), concat_fts, 512, cores, tf.nn.relu, bn, train))
            
            local_fts = tf.concat(points_set, axis=1)
            points = layer_util.sharedMLP_simple ('attn_decoder',local_fts, [256,128,head_pts*3], bn, train)
            points = tf.reshape(points,[self.batch_size, nheads*cores, head_pts, 3])
            coarse = tf.reshape(points,[self.batch_size, self.num_coarse, 3])
            
        with tf.variable_scope ('decoder_fine', reuse=False):
            coarse_pts = tf.reshape(coarse, [1, -1, 3])
            coarse_npts = self.num_coarse * np.ones([self.batch_size],dtype=np.int32)
            
            _,coarse_fts = layer_util.sharedMLP ('coarse_feature',coarse_pts, coarse_npts,[64,128,256], bn, train)
            coarse_fts_exp = tf.tile(tf.expand_dims(coarse_fts, 2), [1,cores*nheads,head_pts,1])
            
            x = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            y = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            grid = tf.meshgrid(x, y)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [self.batch_size, 1024, 1])
            
            global_fts_exp = tf.tile(tf.expand_dims(glb_fts, 2), [1,cores*nheads,head_pts,1])
            concat_fts = tf.concat([coarse_fts_exp,global_fts_exp],axis=3)
            concat_fts = tf.reshape(concat_fts,[self.batch_size, self.num_coarse, 768])
            
            expand_feat = tf.tile(tf.expand_dims(concat_fts, 2), [1, 1, self.grid_size ** 2, 1])
            expand_feat = tf.reshape(expand_feat, [self.batch_size, self.num_fine, 768])
            
            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [self.batch_size, self.num_fine, 3])
            
            feat = tf.concat([grid_feat, point_feat, expand_feat], axis=2)
            
            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [self.batch_size, self.num_fine, 3])
            
            fine = layer_util.sharedMLP_simple('decoder_fine',feat, [512, 256, 128, 3], bn, train) + center
        
        return coarse, fine
    
    def get_loss (self, coarse, fine, gt, gt_fine, alpha):
        
        loss_coarse = tf_util.earth_mover(coarse, gt)
        loss_fine = tf_util.chamfer(fine, gt_fine)
        loss = loss_coarse + alpha * loss_fine
        return loss_coarse, loss_fine, loss
