import os, shutil

rootpath = "slbr-opa-0.4_0.8"
multi_image_path = f"./exp_data/wm_img/{rootpath}/out_test_pics/multi/Reconstruct_image"
watermarked_path = f"./exp_data/wm_img/{rootpath}/Watermarked_image"

all_files = os.listdir(multi_image_path)
print(len(all_files))

starts = [1500, 2250, 5750, 7250, 7500]
whole = []

for s in starts:
    whole += list(range(s, s+250))

for png in all_files:
    whole.remove(int(png[:-4]))

print(whole)
for png in whole:
    shutil.copy(os.path.join(watermarked_path, f"{png}.png"), multi_image_path)

print("Checking..")
all_files = os.listdir(multi_image_path)
starts = [1500, 2250, 5750, 7250, 7500]
whole = []

for s in starts:
    whole += list(range(s, s+250))

for png in all_files:
    whole.remove(int(png[:-4]))

print(whole)