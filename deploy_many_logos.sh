#!/bin/bash
output_dir=$1
logo=$2
pos=$3
siz=$4
ang=$5
opa=$6
noise=$7
random_opa=$8
opa_value=$9
size_value=${10}

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

python deploy_many_logos.py \
--output_dir $log_dir \
--logo_path $logo \
--standard_transform 1 \
--gpu 0,1,2,3 \
--change_pos $pos \
--change_siz $siz \
--change_ang $ang \
--change_opa $opa \
--add_noise $noise \
--random_opa $random_opa \
--opa_value $opa_value \
--size_value $size_value \