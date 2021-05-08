import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import ops
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sample_group_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_sample_group_so.so'))

def farthest_point_sample(inputs, num_core, radius=0.2, min_point=2):
    '''
    inputs: [1,n,3]
    outputs: id:[m], cores:[m,3]
    '''
    return sample_group_module.farthest_point_sample(inputs, npoint=num_core, radius=radius, minnum=min_point)
ops.NoGradient('FarthestPointSample')

def farthest_point_sample_all(npoint,inp):
    '''
    inputs: int32, [1,n,3]
    outputs: id: [1,m]
    '''
    return sample_group_module.farthest_point_sample_all(inp, npoint)
ops.NoGradient('FarthestPointSampleAll')

def sample_group(inputs, num_core, radius=0.2, min_point=6):
    '''
    inputs: [1,n,3]
    outputs: cores: [1,m,3], radiuses: [1,m], local_region: [m,1024,3], cnt[m]
    '''
    return sample_group_module.sample_group(inputs, npoint=num_core, radius=radius, minnum=min_point)
ops.NoGradient('SampleGroup')

if __name__=='__main__':
    import numpy as np   
    import time
    
    point_cloud = np.array([[[0.0,0.0,0.0],[-0.2,0.0,0.0],[0.0,0.25,0.0],[0.2,0.0,0.0],
                             [5.0,5.0,0.0],[5.0,5.6,0.0],[5.0,4.4,0.0],[5.8,5.0,0.0],
                             [10.5,10.5,0.0],[10.0,10.8,0.0]]])
    pc = tf.placeholder(tf.float32, [1, None, 3], 'inputs_part')
    #core_id_all, core_pts_all = farthest_point_sample_all(3, pc)
    core_id, core_pts = farthest_point_sample(pc, 3, radius=0.3, min_point=1)
    cores, radius, local, cnt = sample_group(pc, 3, radius=0.3, min_point=1)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())
    
    output1,output2,output3,output4 = sess.run([cores, radius, local, cnt], feed_dict={pc:point_cloud})
    print (output1)
    print (output2)
    print (output4)
        
        