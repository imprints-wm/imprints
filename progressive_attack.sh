# input="vac0bvmr-no-0.6-size0.5"
input="bvmrpro2-no-0.6-size0.5"

for epoch in 100 200 300 400 500 600 700 800 900 1000 2000;
# for epoch in 1;
do

loop=0
n=$epoch
# mkdir ./out_test_pics/slbr/Reconstruct_image
# mkdir ./exp_data/wm_img/$input/progressive
# mkdir ./exp_data/wm_img/$input/progressive/slbr
# mkdir ./exp_data/wm_img/$input/progressive/slbr/Reconstruct_image

target=${input}_n${n}
mkdir ./exp_data/wm_img/${target}
mkdir ./exp_data/wm_img/${target}/Watermarked_image
mkdir ./exp_data/wm_img/${target}/Watermark
mkdir ./exp_data/wm_img/${target}/Watermark_free_image
mkdir ./exp_data/wm_img/${target}/Mask
mkdir ./exp_data/wm_img/${target}/out_test_pics
mkdir ./exp_data/wm_img/${target}/out_test_pics/bvmr
mkdir ./exp_data/wm_img/${target}/out_test_pics/bvmr/Reconstruct_image

for i in 10 276 514 782 1150 1346 1559 1755 2000 2307 2520 2840 3012 3516 3766 4008 4278 4500 4752 5000;
do
png=${i}.png
cp ./exp_data/wm_img/${input}/Watermarked_image/$png \
    ./exp_data/wm_img/${target}/Watermarked_image
cp ./exp_data/wm_img/${input}/Mask/$png \
    ./exp_data/wm_img/${target}/Mask
cp ./exp_data/wm_img/${input}/Watermark/$png \
    ./exp_data/wm_img/${target}/Watermark
cp ./exp_data/wm_img/${input}/Watermark_free_image/$png \
    ./exp_data/wm_img/${target}/Watermark_free_image
done

python progressive_attack.py --model bvmr \
--model_path ./watermark_removal_works/BVMR/demo_coco/checkpoints/demo_coco/net_baseline_200.pth \
--input_dir ./exp_data/wm_img/${target}/Watermarked_image/ \
--output_dir ./exp_data/wm_img/${target}/out_test_pics/ \
--bgs_dir ./exp_data/wm_img/${target}/Watermark_free_image \
--standard_transform 0 \
--iteration $n

# mv ./exp_data/wm_img/$input/progressive/slbr/Reconstruct_image \
#     ./exp_data/wm_img/$input/progressive/slbr/Reconstruct_image_$(expr $loop + $n)
python metrics.py --eval_path \
./exp_data/wm_img/${target}/ \
> ./exp_data/wm_img/${target}/eval.log

done