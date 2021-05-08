#/bin/bash
/usr/local/cuda-11.1/bin/nvcc tf_sample_group.cu -o tf_sample_group.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF2.4
g++ -std=c++11 tf_sample_group.cpp tf_sample_group.cu.o -o tf_sample_group_so.so -shared -fPIC -I /home/wuhang/anaconda3/envs/py3/lib/python3.7/site-packages/tensorflow/include -I /usr/local/cuda-11.1/include/ -I /home/wuhang/anaconda3/envs/py3/lib/python3.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-11.1/lib64/ -L /home/wuhang/anaconda3/envs/py3/lib/python3.7/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
