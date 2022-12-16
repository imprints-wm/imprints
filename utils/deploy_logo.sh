#!/bin/bash
output_dir=$1
logo=$2

log_dir="./exp_data/wm_img/"$output_dir"/"

if [ ! -d "$log_dir" ];then
    mkdir $log_dir
    echo "创建log文件夹成功"
else
    :
fi

mkdir $log_dir"Mask"
mkdir $log_dir"Watermark"
mkdir $log_dir"Watermark_free_image"
mkdir $log_dir"Watermarked_image"

python deploy_logo.py \
--output_dir $log_dir \
--logo_path $logo \
--standard_transform 1 \
--gpu 0,1,2,3 \
