#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:57:16 2020

@author: wuhang
"""

import tensorflow as tf
import argparse
import numpy as np
import io
import sys
import pcl
sys.path.append("./utils")
import tf_util
import np_util
sys.path.append("./models")
import model
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-model_type', default='fine')
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-num_repeat', type=int, default=1)
parser.add_argument('-num_view', type=int, default=30)

parser.add_argument('-test_path', default='../data/modelnet_test/partial')
parser.add_argument('-coarse_path', default='../data/modelnet_test/full_1k')
parser.add_argument('-fine_path', default='../data/modelnet_test/full_1w')

parser.add_argument('-fine_model_path', default='./restore/fine/')
parser.add_argument('-pretrain_model_path', default='./restore/pretrain/')
parser.add_argument('-pcd_path_coarse', default='./output/pcd_file/coarse/')
parser.add_argument('-pcd_path_fine', default='./output/pcd_file/fine/')
parser.add_argument('-test_version', type=int, default=0)

parser.add_argument('-cores', type=int, default=16)
parser.add_argument('-min_points', type=int, default=6)
parser.add_argument('-nheads', type=int, default=16)
parser.add_argument('-passes', type=int, default=1)
parser.add_argument('-min_dist', type=int, default=0.25)

parser.add_argument('-bn', type=bool, default=False)
parser.add_argument('-in_drop', type=float, default=0.0)
parser.add_argument('-coef_drop', type=float, default=0.0)
parser.add_argument('-reg_coef', type=float, default=0.1)

parser.add_argument('-print_list', action='store_true', default=False)

args = parser.parse_args()

"""
Model parameters
"""
batch_size = 1
inputs_pl = tf.placeholder(tf.float32, [1, None, 3], 'inputs')
npts_pl = tf.placeholder(tf.int32, [batch_size,], 'num_points')
gt_pl = tf.placeholder(tf.float32, [batch_size, 1024, 3], 'ground_truths')
gt_fine_pl = tf.placeholder(tf.float32, [batch_size, 16384, 3], 'ground_truth_fine')

test_loss = tf.placeholder(tf.float32, shape=[])
test_loss_coarse = tf.placeholder(tf.float32, shape=[])
test_loss_fine = tf.placeholder(tf.float32, shape=[])

ball_index_pl = tf.placeholder(tf.int32, shape=[None,])
ball_size_pl = tf.placeholder(tf.int32, shape=[batch_size*args.cores*args.passes,])

train_pl = tf.placeholder(tf.bool, shape=[])

MODEL = model.RAN_1l(inputs_pl, npts_pl, args.cores*args.passes, args.nheads, 1.0, ball_index_pl, ball_size_pl,
                     gt_pl, gt_fine_pl, args.bn, train_pl)

pc_coarse = MODEL.coarse
pc_fine = MODEL.fine
loss_coarse = MODEL.loss_coarse
loss_fine = MODEL.loss_fine
loss = MODEL.loss

tvars = tf.trainable_variables()
tf_util.printvar (tvars)
c_vars = [var for var in tvars if not 'decoder_fine' in var.name]
f_vars = [var for var in tvars if 'decoder_fine' in var.name]

reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss_reg = args.reg_coef * tf.cast(train_pl,tf.float32)*tf.reduce_mean(reg)

dataset,buffer_size = tf_util.prepare_2out(args.test_path,args.coarse_path,args.fine_path,
                                           batch_size,num_view=args.num_view,shuffle=False)
dataset = dataset.repeat(args.num_repeat)
iterator = dataset.make_one_shot_iterator()
test_data, label1, label2 = iterator.get_next()
    

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(var_list=tvars, max_to_keep=1)

print 'Restoring fine model...'
saver.restore(sess, tf.train.latest_checkpoint(args.fine_model_path))

print "------ Test Process ------"    
for j in range(buffer_size):
	
	cloud_list, coarse_list, fine_list = sess.run([test_data, label1, label2])
	inputs, npts, gt, gt_fine  = np_util.read_dataset(cloud_list, coarse_list, fine_list)
	ball_index,ball_size,_,_ = np_util.batch_cluster(inputs, npts, args.cores, args.passes, args.min_dist, args.min_points)
	print "Cluster Done!"
		
	feed_dict={inputs_pl:inputs,npts_pl:npts,ball_index_pl:ball_index,ball_size_pl:ball_size,
               gt_pl:gt,gt_fine_pl:gt_fine,train_pl:False}
	fout,floss,rloss = sess.run([pc_fine,loss_fine,loss_reg],feed_dict=feed_dict)
	
	pcd_name = cloud_list[0]
	pcd_name = pcd_name.split('/')
	pcd_name = pcd_name[-1]
	
	if args.print_list==True:
		print "Test for: " + pcd_name
		print "Fine file: "+coarse_list[0] + ", " + fine_list[0]
		
	print "Fine loss is "+ str(floss) + ", reg: " + str(rloss)
	pcd_out = pcl.PointCloud(16384)
	pcd_out.from_array(fout[0])
	
	save_name = args.pcd_path_fine + str(args.test_version) + '_' + pcd_name
	pcl.save(pcd_out, save_name)
    
print "Test Done!"
sess.close()
