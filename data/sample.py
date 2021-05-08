# coding: utf-8


import open3d as o3d
import numpy as np
import os
import sys


def voxel_downsample(pcd, tg_size = 3000, vs=0.01, delta=0.001, delta_c=5e-7):
    if (np.shape(pcd.points)[0]<tg_size):
        print "Error: Original point cloud size is smaller than target size"
        return 0,0
    pcd_start = pcd.voxel_down_sample(voxel_size=vs)
    if (np.shape(pcd_start.points)[0]>tg_size):        
        while(1):
            downpcd = pcd.voxel_down_sample(voxel_size=vs)
            if (np.shape(downpcd.points)[0]<tg_size):
                break;
            else:
                vs+=delta
        near = np.shape(pcd.voxel_down_sample(voxel_size=vs-delta).points)[0]
        for vss in np.arange(vs-delta,vs+delta_c,delta_c):
            downpcd = pcd.voxel_down_sample(voxel_size=vss)
            if (np.shape(downpcd.points)[0]-tg_size>=0 & np.shape(downpcd.points)[0]<near):
                near = np.shape(downpcd.points)[0]
                vsf=vss
    else:
        while(1):
            downpcd = pcd.voxel_down_sample(voxel_size=vs)
            if (np.shape(downpcd.points)[0]>tg_size):
                break;
            else:
                vs-=delta  
        near = np.shape(pcd.voxel_down_sample(voxel_size=vs).points)[0]
        for vss in np.arange(vs,vs+delta+delta_c,delta_c):
            downpcd = pcd.voxel_down_sample(voxel_size=vss)
            if (np.shape(downpcd.points)[0]-tg_size>=0 & np.shape(downpcd.points)[0]<near):
                near = np.shape(downpcd.points)[0]
                vsf=vss
    return vsf,near




if __name__ == '__main__':
    filename = sys.argv[1]
    tg_size = int(sys.argv[2])
    tg_path = sys.argv[3]
    log_path = sys.argv[4]
    print "Load poincloud from " + filename
    pcd = o3d.io.read_point_cloud(filename)
    #o3d.visualization.draw_geometries([pcd])
    vsf,near = voxel_downsample(pcd,tg_size)
    pcd_filtered = pcd.voxel_down_sample(voxel_size=vsf)
    print "Filtered poincloud number: "+ str(np.shape(pcd_filtered.points)[0])
    #o3d.visualization.draw_geometries([pcd_filtered])
    filtered = np.asarray(pcd_filtered.points)
    for i in range (20):
        np.random.shuffle(filtered)
    cloud_out = filtered[0:tg_size,:]
    print "Adjusted poincloud shape: "+ str(np.shape(cloud_out))
    pcd_save = o3d.geometry.PointCloud()
    pcd_save.points = o3d.utility.Vector3dVector(cloud_out)
    #o3d.visualization.draw_geometries([pcd_save])
    o3d.io.write_point_cloud(tg_path, pcd_save)
    with open(log_path, 'a') as f:
        f.write("Load poincloud from " + filename + "\n")
        f.write("Filtered poincloud number: "+ str(np.shape(pcd_filtered.points)[0])+"\n")
        f.write("Adjusted poincloud shape: "+ str(np.shape(cloud_out)) + "\n")



