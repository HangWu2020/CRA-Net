# CRA-Net

Codes of "Point cloud completion using multiscale feature fusion and cross-regional attention"

The codes in ```code_v1``` are based on ```Tensorflow 1.14```, we will further optimize our codes and make it adapt to ```Tensorflow 2.4.1``` (as a back up for NVIDIA Ampere users) in ```code_v2```.

## Data generation
### Requirements
```
pcl 1.8
python 2.7 
open3d 0.9.0 
python_pcl
VTK
```

### Generation steps 
```
cd data
Compile the C++ files in /genpc and /virtual_cam
bash data_gen.sh -o obj/file/direction/in/your/computer/ -d distance -n num_sample -p pcd/file/direction/you/want/to/save/
```
Please note that the 3D files in ModelNet40 are in ```off``` format, you may need to first transfer and normalize them to ```obj``` files that are readable for ```VTK```, we also provide a transfer script in this folder
```
python format.py off/file/direction/in/your/computer/ obj/file/direction/you/want/to/save/
```
For any questions, or request of our processed models, please contact wuhang100@sjtu.edu.cn.

## Compile CD/EMD
The ```pc_distance``` folder in ```code_v1``` or ```code_v2``` are CD and EMD loss borrowed from [PCN](https://github.com/wentaoyuan/pcn), please follow their instructions to compile them.

## Code V1 
### Requirements
```
python 2.7 
tensorflow 1.14 
open3d 0.9.0 
python_pcl
```
### Train
```
cd code_v1
mkdir log
mkdir -p restore/pretrain restore/fine
python train_cd_emd.py
```
### Test
```
mkdir -p output/pcd_file/coarse output/pcd_file/fine
python test.py
```

## Code V2 
### Requirements
```
python 3.7 
tensorflow 2.4.1
open3d 0.9.0
```
### Train
```
cd code_v2
python train_cd_emd.py
```
## Faster ANQ
### Requirements
```
python 3.7 
tensorflow 2.4.1 (should also work with tf1)
```
We also provide GPU version of ```ANQ```, please refer to ```code_v2/tf_ops``` for compiling. At current step, this customized operation works when ```batchsize=1``` in test.