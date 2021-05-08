#!/bin/bash
format=".obj"
while getopts ":o:p:n:d:h:" opt
do
    case $opt in
    o)
        dir=$OPTARG;;
    p)
        pcd=$OPTARG;;
    n)
        num=$OPTARG;;
    d)
        dis=$OPTARG;;
    h)
        echo "Format: bash data_gen.sh -o obj-dir/ -d distance -n num_sample -p pcd-dir/";;
    ?)
        echo "Enter bash data_gen.sh -help for help";;
    esac
done

if [ ! -d ${pcd} ]
then
      mkdir -p $pcd/full/log/
      mkdir -p $pcd/partial/log/
fi


bash laser_scan.sh -o ${dir} -p ${pcd}partial/ -d ${dis} -s ${num} -l ${pcd}partial/log/
bash scanpc_full.sh -o ${dir} -p ${pcd}full/ -l ${pcd}full/log/
