#!/bin/bash

for logo in {1..20};
do
# rm ./wm_adv_0.png
# rm ./wm_adv_0.pt
# rm ./wm_adv_mask_0.pt
# rm ./exp_data/slbr/wm_adv_0.pt
# rm ./exp_data/slbr/wm_adv_best.png
# rm ./exp_data/slbr/wm_adv_latest.png
# rm ./exp_data/slbr/wm_adv_mask_end.pt
# rm ./exp_data/bvmr/wm_adv_0.pt
# rm ./exp_data/bvmr/wm_adv_best.png
# rm ./exp_data/bvmr/wm_adv_latest.png
# rm ./exp_data/bvmr/wm_adv_mask_end.pt


time=$(date "+%Y-%m-%d-%H-%M-%S")
MODELS=("bvmr" "slbr" "split")
model=$1
# logo=$2

echo "${MODELS[@]}" | grep -wq "$model" &&  echo "testing "$model || exit 1

log_dir="./exp_data/"$model"/logo_log_pro/"$logo"_"$time"/"

if [ ! -d "$log_dir" ];then
    mkdir $log_dir
    echo "创建log文件夹成功"
else
    :
fi

log_name=$log_dir"/train.log"
echo "==> logFile saved in "$log_name

if [ "$model" == "bvmr" ];then
    python train_logo.py --model bvmr \
    --model_path ./watermark_removal_works/BVMR/demo_coco/checkpoints/demo_coco/net_baseline_200.pth \
    --transparences 0.3 0.5 0.7 \
    --output_dir ./adv_wm_pics/ \
    --logo_path "./datasets/CLWD/watermark_logo/test_color/"$logo".png" \
    --epoch 50 \
    --standard_transform 1 \
    --eps 0.2 \
    --log $log_dir \
    --gpu 2 \
    --batch_size 10
elif [ "$model" == "slbr" ];then
    python train_logo.py --model slbr \
    --model_path ./watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar \
    --transparences 0.3 0.5 0.7 \
    --output_dir ./adv_wm_pics/ \
    --logo_path "./datasets/CLWD/watermark_logo/test_color/"$logo".png" \
    --epoch 50 \
    --standard_transform 1 \
    --eps 0.2 \
    --log $log_dir \
    --gpu 0,2,3 \
    --batch_size 15
elif [ "$model" == "split" ];then
    python train_logo.py --model split \
    --model_path ./watermark_removal_works/split_then_refine/27kpng_model_best.pth.tar \
    --transparences 0.3 0.5 0.7 \
    --output_dir ./adv_wm_pics/ \
    --logo_path "./datasets/CLWD/watermark_logo/test_color/"$logo".png" \
    --epoch 50 \
    --standard_transform 1 \
    --eps 0.2 \
    --log $log_dir \
    --gpu 0 \
    --batch_size 8    
else
    :
fi

done