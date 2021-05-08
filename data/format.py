# coding: utf-8

import sys
import numpy as np
import os

# offpath = "./test/airplane_0627.off"
# objpath = "./test0.obj"

def off2obj(offpath,objpath):
    if objpath[-3:]!='obj':
        print "Error: Output is not .obj file"
        return
    if offpath[-3:]!='off':
        print "Error: Input is not .off file"
        return    
    with open(offpath,'r') as f:
        lines = f.readlines()
    if lines[0] != "OFF\n":
        print "Error: Input is not .off file"
        return
    num_pt = int(lines[1].split('\n')[0].split(' ')[0])
    line_pt = [2,num_pt+2]
    print line_pt
    obj_v = []
    for i in range(line_pt[0],line_pt[1]):
        obj_v.append('v '+lines[i])
    print 'Points: ' + str(np.shape(obj_v)[0])
    for i in range(line_pt[1],len(lines)):
        m = lines[i].split('\n')[0].split(' ')
        obj_v.append('f '+str(int(m[1])+1)+' '+str(int(m[2])+1)+' '+str(int(m[3])+1)+'\n')
    print 'Surface:' + str(np.shape(obj_v)[0]-num_pt)
    with open(objpath, 'w') as out:
        for i in range(len(obj_v)):
            out.write(obj_v[i])

def off2obj_norm(offpath,objpath):
    if objpath[-3:]!='obj':
        print "Error: Output is not .obj file"
        return
    if offpath[-3:]!='off':
        print "Error: Input is not .off file"
        return    
    with open(offpath,'r') as f:
        lines = f.readlines()
    if lines[0] != "OFF\n":
        print "Error: Input is not .off file"
        return
    num_pt = int(lines[1].split('\n')[0].split(' ')[0])
    line_pt = [2,num_pt+2]
    print line_pt
    obj_v = []
    obj_m = np.zeros([num_pt,3])
    for i in range(line_pt[0],line_pt[1]):
        obj_m[i-2,0] = float(lines[i].split('\n')[0].split(' ')[0])
        obj_m[i-2,1] = float(lines[i].split('\n')[0].split(' ')[1])
        obj_m[i-2,2] = float(lines[i].split('\n')[0].split(' ')[2])
    obj_mean = obj_m-(obj_m.min(axis=0)+obj_m.max(axis=0))/2.0
    obj_norm = obj_mean/max(-np.min(obj_mean),np.max(obj_mean))
    with open(objpath, 'w') as out:
        for i in range(num_pt):
            index = 'v '+str(obj_norm[i,0])+' '+str(obj_norm[i,1])+' '+str(obj_norm[i,2])+'\n'
            out.write(index)           
        for i in range(line_pt[1],len(lines)):
            m = lines[i].split('\n')[0].split(' ')
            out.write('f '+str(int(m[1])+1)+' '+str(int(m[2])+1)+' '+str(int(m[3])+1)+'\n')

if __name__ == "__main__":
    offdir = sys.argv[1]
    objdir = sys.argv[2]
    if not os.path.exists(objdir):
        os.mkdir(objdir)
    allfilelist = os.listdir(offdir)
    allfilelist.sort()
    for i in range (len(allfilelist)):
        offpath = offdir + allfilelist[i]
        objpath = objdir + allfilelist[i][:-4] + '.obj'
        off2obj_norm(offpath,objpath)
        print 'Model '+ allfilelist[i][:-4] + ' transferred.'

