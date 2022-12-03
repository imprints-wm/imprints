img_dir=
mask_dir=
inpaint_dir=/home/public/imprints/inpaint/inpaint_inputs/
inpaint_out_dir=/home/public/imprints/inpaint/output/

cd /home/imprints/Imprints/watermark_removal_works/ZITS_inpainting

python single_image_test.py --path ./ckpt/zits_places2  --config_file ./config_list/config_ZITS_places2.yml \
--GPU_ids '2' \
--img_path $img_dir \
--mask_path $mask_dir \
--save_path $inpaint_out_dir \
--save_inpaint_path $inpaint_dir