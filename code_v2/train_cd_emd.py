# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import argparse
import sys,os,io
import time
import csv
import open3d as o3d

from models.model_tf2 import *
from utils.tf_util_tf2 import *
from utils.np_util_tf2 import *

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=30)
parser.add_argument('-num_repeat', type=int, default=30)
parser.add_argument('-num_view', type=int, default=30)

parser.add_argument('-train_path', default='../dataset/modelnet_train/partial')
parser.add_argument('-train_coarse_path', default='../dataset/modelnet_train/full_1k')
parser.add_argument('-train_fine_path', default='../dataset/modelnet_train/full_1w')
parser.add_argument('-valid_path', default='../dataset/modelnet_valid/partial')
parser.add_argument('-valid_coarse_path', default='../dataset/modelnet_valid/full_1k')
parser.add_argument('-valid_fine_path', default='../dataset/modelnet_valid/full_1w')

parser.add_argument('-train_coarse', action='store_true', default=False)
parser.add_argument('-log_path', default='./log/')
parser.add_argument('-restore', type=bool, default=False)
parser.add_argument('-checkpoint', default='./restore/fine/model')
parser.add_argument('-model_path', default='./restore/fine/')

parser.add_argument('-cores', type=int, default=16)
parser.add_argument('-min_points', type=int, default=6)
parser.add_argument('-nheads', type=int, default=16)
parser.add_argument('-passes', type=int, default=1)
parser.add_argument('-min_dist', type=int, default=0.25)

parser.add_argument('-base_lr', type=float, default=0.0001)
parser.add_argument('-lr_decay', type=bool, default=True)
parser.add_argument('-lr_decay_epoch', type=int, default=15)
parser.add_argument('-lr_decay_rate', type=float, default=0.7)
parser.add_argument('-lr_clip', type=float, default=1e-6)
parser.add_argument('-bn', type=bool, default=False)
parser.add_argument('-reg_coef', type=float, default=0.1)

parser.add_argument('-print_list', action='store_true', default=False)

args = parser.parse_args()

if not os.path.exists(args.log_path):
	os.makedirs(args.log_path)
else:
    print("Log paths exist")
    
if not os.path.exists(args.model_path):
	os.makedirs(args.model_path)
else:
    print("Model paths exist")    

batch_size = args.batch_size

num_instance = len(os.listdir(args.train_coarse_path))
alpha_step = [2*num_instance,4*num_instance,8*num_instance, 16*num_instance]

global_step = tf.Variable(0, trainable=False, name='global_step')
alpha = tf.train.piecewise_constant(global_step, alpha_step, [0.1, 0.3, 1.0, 3.0, 9.0], 'alpha_op')

min_loss = 0.5
train_loss = tf.placeholder(tf.float32, shape=[])
train_loss_coarse = tf.placeholder(tf.float32, shape=[])
train_loss_fine = tf.placeholder(tf.float32, shape=[])
valid_loss = tf.placeholder(tf.float32, shape=[])
valid_loss_coarse = tf.placeholder(tf.float32, shape=[])
valid_loss_fine = tf.placeholder(tf.float32, shape=[])

inputs_pl = tf.placeholder(tf.float32, [1, None, 3], 'inputs')
npts_pl = tf.placeholder(tf.int32, [batch_size,], 'num_points')
gt_pl = tf.placeholder(tf.float32, [batch_size, 1024, 3], 'ground_truth')
gt_fine_pl = tf.placeholder(tf.float32, [batch_size, 16384, 3], 'ground_truth_fine')

ball_index_pl = tf.placeholder(tf.int32, shape=[None,])
ball_size_pl = tf.placeholder(tf.int32, shape=[batch_size*args.cores*args.passes,])

train_pl = tf.placeholder(tf.bool, shape=[])

if args.lr_decay:
    learning_rate = tf.train.exponential_decay(args.base_lr, global_step, args.lr_decay_epoch*num_instance, args.lr_decay_rate,
                                               staircase=True, name='lr')
    learning_rate = tf.maximum(learning_rate, args.lr_clip)
else:
    learning_rate = tf.constant(args.base_lr, name='lr')

MODEL = RAN_1l(inputs_pl, npts_pl, args.cores*args.passes, args.nheads, alpha, ball_index_pl, ball_size_pl, 
               gt_pl, gt_fine_pl, args.bn, train_pl)
pc_coarse = MODEL.coarse
pc_fine = MODEL.fine
loss_coarse = MODEL.loss_coarse
loss_fine = MODEL.loss_fine
loss = MODEL.loss

reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss_reg = args.reg_coef * tf.cast(train_pl,tf.float32)*tf.reduce_mean(reg)

loss_all = loss + loss_reg

if args.bn==True:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_all,global_step=global_step)
else:
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_all,global_step=global_step)
    
tf.summary.scalar("train_loss", train_loss, collections=['train_summary'])
tf.summary.scalar("train_loss_coarse", train_loss_coarse, collections=['train_summary'])
tf.summary.scalar("train_loss_fine", train_loss_fine, collections=['train_summary'])

tf.summary.scalar("valid_loss_coarse", valid_loss_coarse, collections=['valid_summary'])
tf.summary.scalar("valid_loss_fine", valid_loss_fine, collections=['valid_summary'])

train_summary = tf.summary.merge_all('train_summary')
valid_summary = tf.summary.merge_all('valid_summary')

dataset,buffer_size = prepare_2out(args.train_path,args.train_coarse_path,args.train_fine_path,
                                   batch_size,num_view=args.num_view)
dataset = dataset.repeat(args.num_repeat)
iterator = dataset.make_one_shot_iterator()
train_data, label1, label2 = iterator.get_next()

valid_dataset,valid_buffer_size = prepare_2out(args.valid_path,args.valid_coarse_path,args.valid_fine_path,
                                             batch_size,num_view=args.num_view)
valid_dataset = valid_dataset.repeat(args.num_repeat)
valid_iterator = valid_dataset.make_one_shot_iterator()
valid_data, valid_label1, valid_label2 = valid_iterator.get_next()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

tvars = tf.trainable_variables()
#print ("Vars trainable:")
#printvar (tvars)

gvars = tf.global_variables()
saver = tf.train.Saver(var_list=gvars, max_to_keep=2)

print ("Start training:")
writer = tf.summary.FileWriter(args.log_path)

if args.restore:
    print ('Restoring pretrained model...')
    saver.restore(sess, tf.train.latest_checkpoint(args.model_path))
    
for n in range(1,args.num_repeat+1):
    loss_batch = np.array([0.0,0.0,0.0])
    for j in range(buffer_size//batch_size):
        cloud_list, coarse_list, fine_list = sess.run([train_data, label1, label2])
        inputs, npts, gt, gt_fine = read_dataset(cloud_list, coarse_list, fine_list)
        ball_index,ball_size,_,_ = batch_cluster(inputs, npts, args.cores, args.passes, args.min_dist, args.min_points)
        
        print ("Cluster Done!")
        
        feed_dict={inputs_pl:inputs,npts_pl:npts,ball_index_pl:ball_index,ball_size_pl:ball_size,
                   gt_pl:gt,gt_fine_pl:gt_fine,train_pl:True}
        
        step, alp = sess.run([global_step,alpha])
        closs,floss,rloss,tloss,_ = sess.run([loss_coarse,loss_fine,loss_reg,loss_all,train_op],feed_dict=feed_dict)
                
        print ("Train iter: %d.%d  Global step: %d  aplha:%.2f" %(n,j,step,alp))
        print ("Loss: %.3f, EMD*100: %.3f, CD*1000: %.3f"%(tloss, closs*100, floss*1000))
        print ("L2 Regularization: %.6f"%rloss)
        
        loss_batch = loss_batch + np.array([tloss,closs,floss])
        
        if args.print_list==True:
            print ("Point cloud list for this epoch is:")
            print (cloud_list)
            print (coarse_list)
            print (fine_list)
            
    print ("*********")
    print ("Train iterator %d done"%n)
    loss_avg = loss_batch/buffer_size*batch_size
    print ("Averge Loss: %.3f, EMD*100: %.3f, CD*1000: %.3f"%(loss_avg[0], loss_avg[1]*100, loss_avg[2]*1000))
    train_summary_group = sess.run(train_summary,feed_dict={train_loss:loss_avg[0],train_loss_coarse:loss_avg[1],train_loss_fine:loss_avg[2]})
    writer.add_summary(train_summary_group, n)
    saver.save(sess, args.checkpoint, global_step=n)
    
    if n>15:
        print ("------ Valid Process ------")
        loss_batch = np.array([0.0,0.0])
        for i in range(valid_buffer_size//batch_size):
            cloud_list, coarse_list, fine_list = sess.run([valid_data, valid_label1, valid_label2])
            inputs, npts, gt, gt_fine = read_dataset(cloud_list, coarse_list, fine_list)
            ball_index,ball_size,_,_ = batch_cluster(inputs, npts, args.cores, args.passes, args.min_dist, args.min_points)
            
            print ("Cluster Done!")
            
            feed_dict={inputs_pl:inputs,npts_pl:npts,ball_index_pl:ball_index,ball_size_pl:ball_size,
                       gt_pl:gt,gt_fine_pl:gt_fine,train_pl:False}
            
            closs,floss,rloss = sess.run([loss_coarse,loss_fine,loss_reg],feed_dict=feed_dict)
            step, alp = sess.run([global_step,alpha])
            
            print ("Valid iter: %d.%d  Global step: %d  aplha:%.2f" %(n,i,step,alp))
            print ("EMD*100: %.3f, CD*1000: %.3f"%(closs*100, floss*1000))
            print ("L2 Regularization: %.6f"%rloss)
            
            loss_batch = loss_batch + np.array([closs,floss])
        
        print ("*********")
        print ("Valid iterator %d done"%n)
        loss_avg = loss_batch/valid_buffer_size*batch_size
        print ("Averge EMD*100: %.3f, CD*1000: %.3f"%(loss_avg[0]*100, loss_avg[1]*1000))
        valid_summary_group = sess.run(valid_summary,feed_dict={valid_loss_coarse:loss_avg[0],valid_loss_fine:loss_avg[1]})
        writer.add_summary(valid_summary_group, n)

print ("Done!")
sess.close()