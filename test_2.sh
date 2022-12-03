
# input="/home/public/imprints/exp_data/wm_img/nai-no-0.6/"
# input="/home/public/imprints/exp_data/wm_img/rn-no-0.6/"
input="/home/public/imprints/exp_data/wm_img/slbr-no-0.6/"

mkdir $input/out_test_pics/slbr_with_real_mask
mkdir $input/out_test_pics/slbr_with_real_mask/slbr
mkdir $input/out_test_pics/slbr_with_real_mask/slbr/Mask
mkdir $input/out_test_pics/slbr_with_real_mask/slbr/Watermark
mkdir $input/out_test_pics/slbr_with_real_mask/slbr/Reconstruct_image
mkdir $input/out_test_pics/slbr_with_real_mask/slbr/Watermark_free_image


python test_2.py --model slbr \
--model_path /home/imprints/Imprints/watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar \
--input_dir $input/Watermarked_image \
--output_dir $input/out_test_pics/slbr_with_real_mask/ \
--bgs_dir $input/Watermark_free_image \
--standard_transform 1

python metrics.py --eval_path \
$input/ \
> $input/eval_nonblind.log