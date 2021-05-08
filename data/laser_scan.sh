#!/bin/bash
format=".obj"
while getopts ":o:p:d:s:l:h:" opt
do
    case $opt in
    o)
        dir=$OPTARG;;
    p)
        pcd=$OPTARG;;
    d)
	dis=$OPTARG;;
    s)
	sam=$OPTARG;;
    l)
	log=$OPTARG;;
    h)
        echo "Format: bash laser_scan.sh -o obj-dir/ -p pcd-dir/obj_ -d distance(5.5) -s number_sample(30) -l logdir/";;
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
    if [ "$size" -gt "10000000" ]
    then
        echo "The file is too large"
    else
        ./virtual_cam/build/virtual_cam -modelpath ${filename}.obj -savepath ${pcd}${filename:${dirlength}}_ -camdis ${dis} -num_sample ${sam} -logfile ${log}log
    fi
done
