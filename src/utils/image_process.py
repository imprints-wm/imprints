import os
import cv2
from matplotlib.pyplot import axis
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torch.nn import functional as F
import torchvision.transforms.functional as TF

# read  single picture as arrary and tensor and normalize it
def load_image(path, gray=False, standard_norm=False):
    """
    output shape: if image is 'gray' then shape is (1,256,256); else shape is (3,256,256)
    output range of normalization: if image is 'gray' then range is [0,1]; else range is [-1,1]
    """
    image = np.array(Image.open(path))
    if gray:
        image = (image / 255).astype(np.float32)
        if len(image.shape) == 3:
            image = image[:, :, 0]
            image = np.expand_dims(image, 0)
    else:
        if not standard_norm:
            image = (image / 127.5 - 1).astype(np.float32)
        else:
            image = (image / 255.0).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
    return image, torch.from_numpy(image)


# 读取相关张量数据
def read_tensor(data_item, standard_transform, add_dim=True):
    _, mask = load_image(data_item["mask"], standard_norm=standard_transform, gray=True)
    _, bg = load_image(data_item["bg"], standard_norm=standard_transform)
    _, wm = load_image(data_item["wm"], standard_norm=standard_transform)
    im_np, im_ts = load_image(data_item["im"], standard_norm=standard_transform)

    if add_dim:
        im_ts, mask, bg, wm = (
            im_ts.unsqueeze(0),
            mask.unsqueeze(0),
            bg.unsqueeze(0),
            wm.unsqueeze(0),
        )
    return im_ts, bg, mask, wm


# Save the pictures
def save_adv_pic(path, image_lis, mask_lis, standard_transform, bg_lis, filename_list=None, train=True):
    if train:
        wm_pic_dir = path + "Watermarked_image"
    else:
        wm_pic_dir = path + "Reconstruct_image"
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(path + "Mask")
        os.mkdir(path + "Watermark")
        os.mkdir(wm_pic_dir)
        os.mkdir(path + "Watermark_free_image")
    print("==> current working directory:", os.getcwd())
    print("==> Saving adversarial pictures...")
    for i in range(image_lis.shape[0]):
        img = image_lis[i]
        bg = bg_lis[i]
        wm = mask_lis[i][0]

        if not standard_transform:
            img = image_lis[i] / 2 + 0.5
            bg = bg_lis[i] / 2 + 0.5
            wm = mask_lis[i][0] / 2 + 0.5
        if train:
            bg, wm, img, mask = (
                bg.cpu().numpy(),
                wm.cpu().numpy(),
                img.cpu().numpy(),
                mask_lis[i][1].cpu().numpy(),
            )
        else:
            bg, wm, img, mask = (
                bg.cpu().detach().numpy(),
                wm.cpu().detach().numpy(),
                img.cpu().detach().numpy(),
                mask_lis[i][1].cpu().detach().numpy(),
            )

        bg, wm, img = (
            np.transpose(bg, (1, 2, 0)) * 255,
            np.transpose(wm, (1, 2, 0)) * 255,
            np.transpose(img, (1, 2, 0)) * 255,
        )
        bg, wm, img = (
            bg[:, :, [2, 1, 0]],
            wm[:, :, [2, 1, 0]],
            img[:, :, [2, 1, 0]],
        )  # RGB to BGR !!!
        if filename_list:
            cv2.imwrite(wm_pic_dir + "/" + os.path.basename(filename_list[i]), img)
            cv2.imwrite(path + "Watermark_free_image/" + os.path.basename(filename_list[i]), bg)
            cv2.imwrite(path + "Watermark/" + os.path.basename(filename_list[i]), wm)
        else:
            cv2.imwrite(wm_pic_dir + "/" + str(i) + '.png', img)
            cv2.imwrite(path + "Watermark_free_image/" + str(i) + '.png', bg)
            cv2.imwrite(path + "Watermark/" + str(i) + '.png', wm)

        mask = np.transpose(mask, (1, 2, 0)) * 255
        mask = mask[:, :, [2, 1, 0]]

        if filename_list:
            cv2.imwrite(path + "Mask/" + os.path.basename(filename_list[i]), mask)
        else:
            cv2.imwrite(path + "Mask/" + str(i) + '.png', mask)


def transform_pos(ori_wm, ori_mask, device):
    """
    ori_wm: tensor whose shape is (3,256,256)
    ori_mask: tesor whose shape is (1,1,256,256) or (1,256,256)
    """
    if len(ori_mask.shape) == 4:
        ori_mask.squeeze(0)
    ori_wm, ori_mask = ori_wm.cpu().numpy(), ori_mask.cpu().numpy()
    ori_mask = ori_mask.repeat(3, axis=0)
    ori_wm, ori_mask = np.transpose(ori_wm, (1, 2, 0)), np.transpose(
        ori_mask, (1, 2, 0)
    )
    rows, cols = ori_wm.shape[:2]
    tx = random.randint(1, rows // 32)
    ty = random.randint(1, cols // 32)
    moving_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    wm_moved, mask = cv2.warpAffine(
        ori_wm, moving_matrix, (cols, rows)
    ), cv2.warpAffine(ori_mask, moving_matrix, (cols, rows))
    wm_moved, mask = np.transpose(wm_moved, (2, 0, 1)), np.transpose(mask, (2, 0, 1))

    return torch.from_numpy(wm_moved).to(device), torch.from_numpy(mask).to(device)

def shrink_img(img_ts, size_proportion, standard_transform):
    h, w = img_ts.shape[1:]
    h_resize, w_resize = h * size_proportion[0], w * size_proportion[1]

    res = T.Resize(size=[int(h_resize), int(w_resize)])(img_ts)
    padding_l = (w - w_resize) // 2
    padding_r = w - w_resize - padding_l + 1 
    padding_u = (h - h_resize) // 2 
    padding_d = h - h_resize - padding_u + 1
    val = 0 if standard_transform == 1 else -1
    pad = torch.nn.ConstantPad2d(padding=(int(padding_l), int(padding_r), int(padding_u), int(padding_d)), value=val)
    return pad(res)

def rotate_img(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[1:]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def rotate_img_ts(img_ts, angle, standard_transform):
    fill_val = 0 if standard_transform == 1 else -1
    rotated_img = TF.affine(img_ts, angle=angle, translate=(0,0), scale=1.0, fill=fill_val, shear=0)
    return rotated_img

def translation_img_ts(img_ts,position,standard_transform):
    fill_val = 0 if standard_transform == 1 else -1
    ret = TF.affine(img_ts, angle=0, translate=position, scale=1.0, fill=fill_val, shear=0)
    return ret

def deploy_wm(
    wm, mask, bg, opacity, size_proportion, position, rotation_angle, standard_transform
):
    """
                 wm: tensor whose shape is (3, height, width) 
                 bg: similar to wm
            opacity: num in range [0,1]
    size_proportion: tuple of proportions -> (height_proportion[0,1], width_proportion[0,1])
           position: tuple of moved steps (tx, ty)
    `rotation_angle:  

    return =>
    ultimate_im_ts: watermarked image whose shape is (3, height, width)
    ultimate_wm_ts: transformed watermark whose shape is (3, height, width)
    ultimate_mask0: transformed mask whose shape is (3, height, width)
    ultimate_mask1: transformed mask whose shape is (1, height, width)
    """ 
    wm = shrink_img(wm, size_proportion, standard_transform)
    mask = shrink_img(mask.repeat(3,1,1),size_proportion,standard_transform=1)

    moved_wm = translation_img_ts(img_ts=wm,position=position,standard_transform=standard_transform)
    moved_mask = translation_img_ts(img_ts=mask,position=position,standard_transform=1)

    rotate_wm = rotate_img_ts(img_ts=moved_wm,angle=rotation_angle,standard_transform=standard_transform)
    rotate_mask = rotate_img_ts(img_ts=moved_mask,angle=rotation_angle,standard_transform=1)

    ultimate_wm = rotate_wm
    ultimate_mask_ts = rotate_mask
    ultimate_im_ts = (
        ultimate_wm.cuda() * ultimate_mask_ts * (1 - opacity)
        + bg * (1 - ultimate_mask_ts)
        + bg * ultimate_mask_ts * opacity
    )

    ultimate_wm_ts = ultimate_wm

    return ultimate_im_ts, ultimate_wm_ts, ultimate_mask_ts.cuda(), ultimate_mask_ts[0].unsqueeze(0).cuda()


