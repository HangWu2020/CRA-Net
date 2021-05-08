#!/bin/bash

format=".obj"
while getopts ":o:p:l:h:" opt
do
    case $opt in
    o)
        dir=$OPTARG;;
    p)
        pcd=$OPTARG;;
    l)
        log=$OPTARG;;
    h)
        echo "Format: bash pcl_sample.sh -o obj-dir/ -p pcd-dir/ -l log-dir/";;
    ?)
        echo "Enter bash scanpc_full.sh -help for help";;
    esac
done
dirlength=${#dir}
if [ ! -d ${pcd} ]
then
      mkdir -p $pcd
fi

if [ ! -d ${log} ]
then
      mkdir -p $log
fi

for f in $(ls ${dir}*${format})
do
    filename=${f%.*}
    echo ${filename}
    size=`ls -l ${f} | awk '{print $5}'`
    echo "Size of file: $size"
    if [ "$size" -gt "200000000" ]
    then
        echo "The file is too large"
    else

	pcl_mesh_sampling ${filename}.obj ${pcd}${filename:${dirlength}}_1w.pcd -n_samples 16384 -leaf_size 0.0002 -no_vis_result 
	python sample.py ${pcd}${filename:${dirlength}}_1w.pcd 1024 ${pcd}${filename:${dirlength}}_1k.pcd ${log}log

        #./genpc/build/genpc ${filename}.obj ${pcd}${filename:${dirlength}}.pcd ${log}log
        #python sample.py ${pcd}${filename:${dirlength}}.pcd 16384 ${pcd}${filename:${dirlength}}_1w.pcd ${log}log
        #python sample.py ${pcd}${filename:${dirlength}}_1w.pcd 1024 ${pcd}${filename:${dirlength}}_1k.pcd ${log}log
    fi
done

echo Done!
