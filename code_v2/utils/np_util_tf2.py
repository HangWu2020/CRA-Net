# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

def batch_cluster(cloud_full, npts, cores, passes, min_dist, min_points=6):
    """
    Inputs: cloud [1,None,3]
    """
    cloud_full = np.squeeze(cloud_full)
    batch_size = np.shape(npts)[0]
    ball_index_all = []
    ball_size_all = []
    center_all = []
    r_all = []
    
    for i in range(0,batch_size):
        cloud = cloud_full[np.sum(npts[:i]):np.sum(npts[:i+1]),0:3]
        ball_index,ball_size,center,r = fps_knn_ritter(cloud, cores,passes, min_dist, min_points)
        
        ball_index = ball_index + np.sum(npts[:i])
        ball_index_all.append(ball_index)
        ball_size_all.append(ball_size)
        center_all.append(center)
        r_all.append(r)
        
    ball_index_all = np.hstack(ball_index_all)
    ball_size_all = np.hstack(ball_size_all)
    center_all = np.vstack(center_all)
    r_all = np.hstack(r_all)
   
    return ball_index_all,ball_size_all,center_all,r_all

def fps_knn_ritter (cloud, cores, passes, min_dist=0.25, min_points=6):
    """
    Inputs: cloud [pts,3]
    """
    num_pts = np.shape(cloud)[0]
    key_index_all = []
    key_distance = 12.0*np.ones([num_pts],dtype=np.float32)        
    key_index = np.random.randint(0, num_pts)
    
    i = 0
    while i < cores*passes:
        dist = np.sum((cloud[:,0:3]-cloud[key_index,0:3])**2,-1)        
        ball_size = np.sum((dist <= (min_dist**2)).astype(int))
        
        if ball_size >= min_points:
            key_index_all.append(key_index)
            i += 1
            
        mask = (dist < key_distance).astype(float)
        key_distance = key_distance - np.multiply(mask,key_distance) + np.multiply(mask,dist)
        key_index = np.argmax(key_distance)
           
    key_index_all = np.hstack(key_index_all)
    
    ball_index_all = []
    ball_size_all = []
    center_all = []
    r_all = []
    
    for j in range (passes):
        key_index_1p = key_index_all[j*cores:(j+1)*cores]
        ball_index_1p,ball_size_1p,center_1p,r_1p = knn_ritter(key_index_1p, cloud)
        
        ball_index_all.append(ball_index_1p)
        ball_size_all.append(ball_size_1p)
        center_all.append(center_1p)
        r_all.append(r_1p)
    
    ball_index_all = np.hstack(ball_index_all)
    ball_size_all = np.hstack(ball_size_all)
    center_all = np.vstack(center_all)
    r_all = np.hstack(r_all)
    
    return ball_index_all,ball_size_all,center_all,r_all 

def fps_knn_ritter_2 (cloud, cores, passes=1, min_dist=0.25, min_points=0):
    """
    Inputs: cloud [pts,3]
    """
    num_pts = np.shape(cloud)[0]
    ball_index_npass = []
    ball_size_npass = []
    center_npass = []
    r_npass = []
    
    for n in range (passes):
        key_index_all = []
        key_distance = 12.0*np.ones([num_pts],dtype=np.float32)        
        key_index = np.random.randint(0, num_pts)

        i = 0
        while i < cores:
            dist = np.sum((cloud[:,0:3]-cloud[key_index,0:3])**2,-1)        
            ball_size = np.sum((dist <= (min_dist**2)).astype(int))

            if ball_size >= min_points:
                key_index_all.append(key_index)
                i += 1

            mask = (dist < key_distance).astype(float)
            key_distance = key_distance - np.multiply(mask,key_distance) + np.multiply(mask,dist)
            key_index = np.argmax(key_distance)

        key_index_all = np.hstack(key_index_all)
        ball_index_all,ball_size_all,center_all,r_all = knn_ritter(key_index_all, cloud)
        
        ball_index_npass.append(ball_index_all)
        ball_size_npass.append(ball_size_all)
        center_npass.append(center_all)
        r_npass.append(r_all)
        
    ball_index_npass = np.hstack(ball_index_npass)
    ball_size_npass = np.hstack(ball_size_npass)
    center_npass = np.vstack(center_npass)
    r_npass = np.hstack(r_npass)
    
    return ball_index_npass,ball_size_npass,center_npass,r_npass


def knn_ritter(key_index_all, cloud):
    dist_all = []
    ball_index_all = []
    ball_size_all = []
    
    center_all = []
    r_all = []
    
    for i in (key_index_all):
        dist = np.sum((cloud-cloud[i])**2,-1)
        dist_all.append(dist)
    dist_all = np.vstack(dist_all)
    index = np.argmin(dist_all,axis=0)
    for i in range(len(key_index_all)):
        ball_index = np.where(index==i)[0]
        center, r = ritter_sphere(cloud[ball_index])
        r = np.max([r,0.2])
        center_all.append(center)
        r_all.append(r)
        
        ball_index = ball_query(cloud, center, r)        
        ball_index_all.append(ball_index)
        ball_size_all.append(len(ball_index))
        
    ball_index_all = np.hstack(ball_index_all)
    ball_size_all = np.hstack(ball_size_all)        
    center_all = np.vstack(center_all)
    r_all = np.hstack(r_all)
      
    return ball_index_all,ball_size_all,center_all,r_all


def ritter_sphere(points):
    max_xyz = np.argmax(points,axis=0)
    min_xyz = np.argmin(points,axis=0)
    a0 = np.array([np.sum((points[max_xyz[0]]-points[min_xyz[0]])**2),
                   np.sum((points[max_xyz[1]]-points[min_xyz[1]])**2),
                   np.sum((points[max_xyz[2]]-points[min_xyz[2]])**2)])
    max_axis = np.argmax(a0)
    
    pts = points[[max_xyz[max_axis],min_xyz[max_axis]]]
    center = (pts[0]+pts[1])/2.0
    r = np.sqrt(np.sum((pts[0]-pts[1])**2)/4.0)
    
    pt_dist = np.sqrt(np.sum((points-center)**2,axis=1))
    max_dist = np.max(pt_dist)
    
    while max_dist-r > 0:
                
        new_pt = points[np.argmax(pt_dist)]
        k = (max_dist-r)/(2.0*max_dist)
        center = center + (new_pt-center)*k
        r = r + (max_dist-r)/1.8
        
        pt_dist = np.sqrt(np.sum((points-center)**2,axis=1))
        max_dist = np.max(pt_dist)
        
    return center, r

def ball_query(cloud, center, r):
    dist = np.sum((cloud-center)**2,axis=1)  
    ball_index = np.where(dist<=(r**2))[0]
    return ball_index

def read_dataset(cloud_list, coarse_list, fine_list):
    npts = []
    pcd = []
    for p in cloud_list:
        pc_batch = read_pcd(p)
        inpts = np.shape(pc_batch)[0]
        npts.append(inpts)
        pcd.append(pc_batch)
    npts = np.array(npts)
    pcd = np.vstack(pcd)
    pcd = np.expand_dims(pcd, axis=0)
    
    gt_pc = []
    for p in coarse_list:
        gt_batch = read_pcd(p)
        gt_pc.append(gt_batch)
    gt_pc = np.reshape(gt_pc,[-1,1024,3])

    gt_pc_fine = []
    for pf in fine_list:
        gt_batch_fine = read_pcd(pf)
        gt_pc_fine.append(gt_batch_fine)
    gt_pc_fine = np.reshape(gt_pc_fine,[-1,16384,3])

    return pcd, npts, gt_pc, gt_pc_fine

def read_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    pc_array = np.asarray(pcd.points, dtype=np.float32)
    return pc_array

def save_pcd(pc_array,path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array)
    o3d.io.write_point_cloud(path, pcd)
    
