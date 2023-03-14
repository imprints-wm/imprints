import os
import random

# Obtain pictures' path
def collect_synthesized(_source, is_sort=True):
    paths = []
    for root, _, files in os.walk(_source):
        if is_sort:
            files.sort()
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if (
                file_extension == ".png"
                or file_extension == ".jpg"
                or file_extension == ".jpeg"
            ) and (
                "real" not in file_name
                and "reconstructed" not in file_name
                and "grid" not in file_name
            ):
                paths.append(os.path.join(root, file))
    if not is_sort:
        random.shuffle(paths)
    return paths

# Get a dictionary list of paths
def get_data_list(wm_path, mask_path, bg_path, im_path, shuffle=False):
    data_lis = []
    Wm = collect_synthesized(wm_path)
    Masks = collect_synthesized(mask_path)
    Targets = collect_synthesized(bg_path)
    inputs = collect_synthesized(im_path)
    for i in range(len(Wm)):
        dic = {}
        dic["wm"] = Wm[i]
        dic["mask"] = Masks[i]
        dic["bg"] = Targets[i]
        dic["im"] = inputs[i]
        data_lis.append(dic)
    if shuffle:
        random.shuffle(data_lis)
    return data_lis

def get_test_list(bg_path, shuffle=False):
    data_lis = []
    Targets = collect_synthesized(bg_path)
    for i in range(len(Targets)):
        dic = {}
        dic["bg"] = Targets[i]
        data_lis.append(dic)
    if shuffle:
        random.shuffle(data_lis)
    return data_lis