#!/bin/bash

rm /home/imprints/Imprints/wm_adv_0.png
rm /home/imprints/Imprints/wm_adv_0.pt
rm /home/imprints/Imprints/wm_adv_mask_0.pt
rm /home/imprints/Imprints/ckpt_wm/slbr/wm_adv_0.pt
rm /home/imprints/Imprints/ckpt_wm/slbr/wm_adv_best.png
rm /home/imprints/Imprints/ckpt_wm/slbr/wm_adv_latest.png
rm /home/imprints/Imprints/ckpt_wm/slbr/wm_adv_mask_end.pt


time=$(date "+%Y-%m-%d-%H-%M-%S")
MODELS=("bvmr" "slbr" "split")
model=$1

echo "${MODELS[@]}" | grep -wq "$model" &&  echo "testing "$model || exit 1

log_dir="./ckpt_wm/"$model"/log/"$time"/"

if [ ! -d "$log_dir" ];then
    mkdir $log_dir
    echo "创建log文件夹成功"
else
    :
fi

log_name=$log_dir"/train.log"
echo "==> logFile saved in "$log_name

if [ "$model" == "bvmr" ];then
    python train.py --model bvmr \
    --model_path ./watermark_removal_works/BVMR/demo_coco/checkpoints/demo_coco/net_baseline_200.pth \
    --transparences 0.3 0.5 0.7 \
    --output_dir ./adv_wm_pics/ \
    --epoch 20 \
    --text Imprints \
    --text_size 45 \
    --standard_transform 0 > $log_name &
elif [ "$model" == "slbr" ];then
    python train.py --model slbr \
    --model_path /home/imprints/Imprints/watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar \
    --transparences 0.3 0.5 0.7 \
    --output_dir ./adv_wm_pics/ \
    --text "S&P" \
    --text_size 80 \
    --epoch 100 \
    --standard_transform 1 \
    --eps 0.2 \
    --log $log_dir \
    --gpu 0,1,2,3 \
    --batch_size 20
elif [ "$model" == "split" ];then
    python train.py --model split \
    --model_path ./watermark_removal_works/split_then_refine/27kpng_model_best.pth.tar \
    --transparences 0.3 0.5 0.7 \
    --batch_size 5 \
    --text Imprints \
    --text_size 45 \
    --output_dir ./adv_wm_pics/ \
    --standard_transform 1 > $log_name &
else
    :
fi