import os, shutil

rootpath = "./exp_data/wm_img/slbr-no-0.6-sample100"
commer = rootpath+"/out_test_pics/commer"

for f in os.listdir(commer):
    # print(f.split('-')[0]+'.png')
    os.rename(
            os.path.join(commer, f),
            os.path.join(commer, f.split('-')[0]+'.png')
            )  