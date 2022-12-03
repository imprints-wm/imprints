input=$1

for wr in slbr bvmr split;
do
mkdir ${input}/out_test_pics
mkdir ${input}/out_test_pics/${wr}
mkdir ${input}/out_test_pics/${wr}/Mask
mkdir ${input}/out_test_pics/${wr}/Reconstruct_image
mkdir ${input}/out_test_pics/${wr}/Watermark
mkdir ${input}/out_test_pics/${wr}/Watermark_free_image
done

### slbr
python test.py --model slbr \
--model_path ./watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar \
--input_dir $input/Watermarked_image \
--output_dir $input/out_test_pics/ \
--bgs_dir $input/Watermark_free_image \
--standard_transform 1

# mkdir ./out_test_pics/slbr/Reconstruct_image

# python test.py --model slbr \
# --model_path ./watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar \
# --input_dir ./out_test_pics/slbr/Reconstruct_image_13_0 \
# --output_dir ./out_test_pics/ \
# --bgs_dir ./adv_wm_pics/slbr/Watermark_free_image \
# --standard_transform 1

# mv ./out_test_pics/slbr/Reconstruct_image ./out_test_pics/slbr/Reconstruct_image_13_1

# exit

# Progressive attack for 10 iterations
# for loop in {51..100}
# do

# mkdir ./out_test_pics/slbr/Reconstruct_image

# python test.py --model slbr \
# --model_path ./watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar \
# --input_dir ./out_test_pics/slbr/Reconstruct_image_${loop} \
# --output_dir ./out_test_pics/ \
# --bgs_dir ./adv_wm_pics/slbr/Watermark_free_image \
# --standard_transform 1

# mv ./out_test_pics/slbr/Reconstruct_image ./out_test_pics/slbr/Reconstruct_image_$(expr $loop + 1)

# done

### bvmr
python test.py --model bvmr \
--model_path ./watermark_removal_works/BVMR/demo_coco/checkpoints/demo_coco/net_baseline_200.pth \
--input_dir $input/Watermarked_image \
--output_dir $input/out_test_pics/ \
--bgs_dir $input/Watermark_free_image \
--standard_transform 0

python test.py --model split \
--model_path ./watermark_removal_works/split_then_refine/27kpng_model_best.pth.tar \
--input_dir $input/Watermarked_image \
--output_dir $input/out_test_pics/ \
--bgs_dir $input/Watermark_free_image \
--standard_transform 1