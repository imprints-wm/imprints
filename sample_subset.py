import os, shutil
image_path = "/home/public/imprints/exp_data/wm_img/rn-no-0.6-order"
watermarked_image_path = os.path.join(image_path, "Watermarked_image")

sample_number = 100

save_path = image_path + f"-sample{sample_number}"
subdir = ['Watermarked_image', 'Watermark', 'Watermark_free_image', 'Mask']
os.mkdir(save_path)

for sd in subdir:
    os.mkdir(os.path.join(save_path, sd))

os.mkdir(os.path.join(save_path, "out_test_pics"))
os.mkdir(os.path.join(
            os.path.join(save_path, "out_test_pics"),
            "commer"
            )
        )


all_files = os.listdir(watermarked_image_path)
print(len(all_files))

sep = len(all_files)//sample_number

for i in range(sample_number):
    for sd in subdir:
        shutil.copy(
            os.path.join(
                os.path.join(image_path, sd), 
                all_files[i*sep]
            ), 
            os.path.join(save_path, sd)
            )
