import random
import torch
import cv2
import os
import numpy as np
from .image_process import read_tensor
from tqdm import tqdm
import time


def process_model_output(outputs, model_name):
    if model_name == "bvmr":
        return outputs[0], outputs[1]
    elif model_name == "slbr":
        return outputs[0][0], outputs[1][0]
    elif model_name == "split":
        return outputs[0][0], outputs[1]


def valid_noise(
    model, model_name, wm, mask, valid_data_lis, transparences, min_val, device
):
    bce = torch.nn.BCELoss()
    masks = [mask for i in range(len(valid_data_lis) * len(transparences))]
    masks = torch.stack(masks, dim=0)
    images = [
        read_tensor(
            valid_data_lis[i],
            standard_transform=(min_val != -1),
            add_dim=False,
        )[1].to(device)
        * ((1 - mask) + mask * transparence)
        + wm * mask * (1 - transparence)
        for transparence in transparences
        for i in range(len(valid_data_lis))
    ]
    image_ori_lis = [
        (
            read_tensor(
                valid_data_lis[i],
                standard_transform=(min_val != -1),
                add_dim=False,
            )[1].to(device)
            * ((1 - mask) + mask * transparence)
            + wm * mask * (1 - transparence)
        )
        .clone()
        .detach()
        for transparence in transparences
        for i in range(len(valid_data_lis))
    ]
    inputs = torch.stack(images, dim=0)
    inputs.requires_grad = False

    outputs = model(inputs)
    guess_images, guess_mask = process_model_output(outputs, model_name)
    model.zero_grad()

    expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
    reconstructed_pixels = guess_images * expanded_guess_mask
    reconstructed_images = inputs * (1 - expanded_guess_mask) + reconstructed_pixels
    pixel_loss = inputs[0] - image_ori_lis[0] + transparences[0] * wm
    refine_loss = torch.abs(reconstructed_images - inputs).mean()
    total_loss = bce(guess_mask, masks) + 0.02 * pixel_loss.mean() + 0.001 * refine_loss
    return total_loss.item()


def build_noise(
    model,
    wm,
    mask,
    data_lis,
    device,
    model_name,
    min_val=-1,
    epoch=1,
    sample_num=40,
    batch_size=10,
    valid_size=5,
    max_iters=200,
    alpha=4 / 255,
    eps=0.5,
    loss_threshold=1.6,
    transparences=[0.3, 0.5, 0.7],
    is_save=False,
):
    dir_path, file_name = "./ckpt_wm/" + model_name, "/wm_adv_0.pt"
    bce = torch.nn.BCELoss()
    model.eval()
    eta = None
    tmp_loss = 0.0
    valid_data_lis = data_lis[sample_num : sample_num + valid_size]
    data_lis = data_lis[:sample_num]
    for epoch_no in range(epoch):
        random.shuffle(data_lis)
        if os.path.exists(dir_path + file_name) and epoch_no!=0:
            del wm
            wm = torch.load(dir_path + file_name,map_location = device)
        for j in range(sample_num // batch_size):
            # 处理masks，以便后面求Loss
            masks = [mask for i in range(batch_size * len(transparences))]
            masks = torch.stack(masks, dim=0)

            if eta is not None and j != 0:
                wm = torch.clamp(wm + eta, min=min_val, max=1)

            images = [
                read_tensor(
                    data_lis[i + j * batch_size],
                    standard_transform=(min_val != -1),
                    add_dim=False,
                )[1].to(device)
                * ((1 - mask) + mask * transparence)
                + wm * mask * (1 - transparence)
                for transparence in transparences
                for i in range(batch_size)
            ]
            # TODO: can be optimized
            image_ori_lis = [
                (
                    read_tensor(
                        data_lis[i + j * batch_size],
                        standard_transform=(min_val != -1),
                        add_dim=False,
                    )[1].to(device)
                    * ((1 - mask) + mask * transparence)
                    + wm * mask * (1 - transparence)
                )
                .clone()
                .detach()
                for transparence in transparences
                for i in range(batch_size)
            ]
            for k in range(max_iters):
                inputs = torch.stack(images, dim=0)
                inputs.requires_grad = True

                outputs = model(inputs)
                guess_images, guess_mask = process_model_output(outputs, model_name)
                model.zero_grad()
                # print('==> ims shape:',guess_images[0].shape)
                # print('==> mask shape:',guess_mask[0].shape)
                expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
                reconstructed_pixels = guess_images * expanded_guess_mask
                reconstructed_images = (
                    inputs * (1 - expanded_guess_mask) + reconstructed_pixels
                )
                pixel_loss = inputs[0] - image_ori_lis[0] + transparences[0] * wm
                refine_loss = torch.abs(reconstructed_images - inputs).mean()
                total_loss = (
                    bce(guess_mask, masks)
                    + 0.02 * pixel_loss.mean()
                    + 0.001 * refine_loss
                )
                total_loss.backward()

                print("==> loss: %.6f" % (total_loss.item()))
                if total_loss.item() > loss_threshold:
                    break

                del outputs

                sign = torch.mean(inputs.grad, dim=0).sign()
                image_adv_lis = [
                    image + alpha * sign * mask for id, image in enumerate(images)
                ]
                eta = torch.mean(
                    torch.stack(
                        [
                            torch.clamp(image_adv - image_ori, min=-eps, max=eps)
                            for image_adv, image_ori in zip(
                                image_adv_lis, image_ori_lis
                            )
                        ],
                        dim=0,
                    ),
                    dim=0,
                )
                images = [
                    torch.clamp(image_ori + eta, min=min_val, max=1).detach_()
                    for image_ori in image_ori_lis
                ]
                del image_adv_lis
                del inputs
                del sign
            del masks
            del images
            del image_ori_lis
        cur_wm = torch.clamp(wm + eta, min=min_val, max=1)
        cur_loss = valid_noise(
            model=model,
            model_name=model_name,
            wm=cur_wm,
            mask=mask,
            valid_data_lis=valid_data_lis,
            transparences=transparences,
            min_val=min_val,
            device=device,
        )
        if cur_loss > tmp_loss:
            tmp_loss = cur_loss
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            torch.save(cur_wm, dir_path + file_name)
            if min_val == -1:
                cur_wm = (cur_wm) / 2 + 0.5
            cur_wm = cur_wm.cpu().numpy()
            cur_wm = np.transpose(cur_wm, (1, 2, 0)) * 255
            cur_wm.astype(np.uint8)
            cv2.imwrite(dir_path + "/wm_adv_0.png", cur_wm)
        print(
            "==> epoch_no: {}  currrent_loss: {:.4f}   max_loss: {:.4f}".format(
                epoch_no, cur_loss, tmp_loss
            )
        )
    torch.save(mask, dir_path + "/wm_adv_mask_0.pt")
    if is_save:
        wm_adv = torch.clamp(wm + eta, min=min_val, max=1).detach_()
        torch.save(wm_adv, "wm_adv_0.pt")
        torch.save(mask, "wm_adv_mask_0.pt")
        # print("==> DEBUG",min_val)
        # print(wm_adv)
        if min_val == -1:
            wm_adv = (wm_adv) / 2 + 0.5
        wm_adv = wm_adv.cpu().numpy()
        wm_adv = np.transpose(wm_adv, (1, 2, 0)) * 255
        wm_adv.astype(np.uint8)
        cv2.imwrite("wm_adv_0.png", wm_adv)

    return torch.clamp(wm + eta, min=min_val, max=1).detach_()
