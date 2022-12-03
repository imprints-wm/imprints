input=$1

### slbr
python adaptive_test.py --model slbr \
--model_path /home/imprints/Imprints/watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar \
--input_dir $input/Watermarked_image \
--output_dir ./out_test_pics/ \
--bgs_dir $input/Watermark_free_image \
--standard_transform 1

### bvmr
python adaptive_test.py --model bvmr \
--model_path ./watermark_removal_works/BVMR/demo_coco/checkpoints/demo_coco/net_baseline_200.pth \
--input_dir $input/Watermarked_image \
--output_dir ./out_test_pics/ \
--bgs_dir $input/Watermark_free_image \
--standard_transform 0

python adaptive_test.py --model split \
--model_path ./watermark_removal_works/split_then_refine/27kpng_model_best.pth.tar \
--input_dir $input/Watermarked_image \
--output_dir ./out_test_pics/ \
--bgs_dir $input/Watermark_free_image \
--standard_transform 1