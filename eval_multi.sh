# for i in nai-no-0.6-order slbr-pos-0.6 slbr-ang-0.6 slbr-siz-0.6 slbr-opa-0.4_0.8;
for i in slbr-no-0.6;
do
python metrics.py \
--eval_path ./exp_data/wm_img/$i \
--stage 5 > ./exp_data/wm_img/$i/eval_multi.log

done