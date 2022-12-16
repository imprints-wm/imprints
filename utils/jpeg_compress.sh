
for kernel in 5 7 9 11;
do

python gaussian_blur.py \
--png ./exp_data/wm_img/slbr-no-0.6/ \
--kernel_size $kernel \

done