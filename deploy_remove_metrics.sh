
for dir in `ls ./exp_data/slbr/logo_log_test/`;
do
save_dir="slbr-all-0.3_0.7-"`echo $dir|cut -d'_' -f1`
wm_dir="./exp_data/slbr/logo_log_test/"$dir
echo $save_dir
echo $wm_dir

bash deploy_logo.sh $save_dir $wm_dir

bash test.sh /home/imprints/Imprints/exp_data/wm_img/$save_dir

python metrics.py --eval_path \
/home/imprints/Imprints/exp_data/wm_img/$save_dir \
> /home/imprints/Imprints/exp_data/wm_img/$save_dir/eval.log

done