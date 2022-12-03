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
    mse = torch.nn.MSELoss(reduction="mean")
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
    inputs = torch.stack(images, dim=0)
    inputs.requires_grad = False

    outputs = model(inputs)
    guess_images, guess_mask = process_model_output(outputs, model_name)
    model.zero_grad()

    expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
    reconstructed_pixels = guess_images * expanded_guess_mask
    reconstructed_images = inputs * (1 - expanded_guess_mask) + reconstructed_pixels
    pixel_loss = wm*mask
    refine_loss = torch.abs(reconstructed_images - inputs).mean()
    total_loss = bce(guess_mask, masks) + 0.0001 * pixel_loss.mean() + 0.001 * refine_loss
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
    max_iters=10,
    alpha=4 / 255,
    eps=0.01,
    loss_threshold=1.6,
    transparences=[0.3, 0.5, 0.7],
    is_save=False,
):
    eta = None
    tmp_loss = 0.0
    dir_path, file_name = "./ckpt_wm/" + model_name, "/wm_adv_0.pt"

    bce = torch.nn.BCELoss(reduction="mean")
    mse = torch.nn.MSELoss(reduction="mean")
    model.eval()

    # validation host images
    valid_data_lis = data_lis[sample_num : sample_num + valid_size]
    # training host images
    data_lis = data_lis[:sample_num]

    # ?
    masks = [mask for _ in range(batch_size * len(transparences))]  # 处理masks，以便后面求Loss
    masks = torch.stack(masks, dim=0)
    
    # initial watermark
    wm_ori = wm.clone().detach()

    for epoch_no in range(epoch):
        random.shuffle(data_lis)
        # save and load the watermark every epoch
        if os.path.exists(dir_path + file_name) and epoch_no != 0:
            wm = torch.load(dir_path + file_name, map_location=device)

        for j in range(sample_num // batch_size):
            # a batch of host images, batchsize * transparences
            # TODO: transparences should not be placed here
            bgs = [
                read_tensor(
                    data_lis[i + j * batch_size],
                    standard_transform=(min_val != -1),
                    add_dim=False,
                )[1].to(device)
                for t in range(len(transparences))
                for i in range(batch_size)
            ]

            for k in range(max_iters):
                # TODO: requires_grad
                #####
                # wm

                # EOT here, (wm, mask)
                # wm - n * t(wm)
                # wm_eot = [wm0, ...] batch wms

                # watermarking (batch), transparences
                images = [
                    # (1 - (1-trans) * mask) * bgs + mask * (1-trans) * wm
                    # (1 - alpha * mask) * bgs + mask * alpha * wm
                    bgs[i] * ((1 - mask) + mask * transparence)
                    + wm * mask * (1 - transparence)
                    for transparence in transparences
                    for i in range(batch_size)
                ]
                inputs = torch.stack(images, dim=0)
                wm.requires_grad = True     # TODO: Shouldn't it be placed before watermarking images?

                outputs = model(inputs)
                guess_images, guess_mask = process_model_output(outputs, model_name)
                model.zero_grad()
                # print('==> ims shape:',guess_images[0].shape)
                # print('==> mask shape:',guess_mask[0].shape)

                # Is it all the watermark removal methods process results in this way?
                # Yes.
                expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
                reconstructed_pixels = guess_images * expanded_guess_mask
                reconstructed_images = (
                    inputs * (1 - expanded_guess_mask) + reconstructed_pixels
                )

                mask_loss = bce(guess_mask, masks)
                # TODO: dont simply maximize the values of wm
                # It contradicts the cliping [-eps, eps] part
                # pixel_loss = (wm * mask).mean()
                pixel_loss = torch.abs(wm - wm_ori).mean()


                # TODO: use l1-norm or l2-norm?
                refine_loss = torch.abs(reconstructed_images - inputs).mean()

                # TODO: a stability loss
                # stability_loss = || jacobian(func, value) ||_p

                total_loss = (
                    -1 * mask_loss + 
                    0.0001 * pixel_loss + 
                    0.001 * refine_loss
                )
                total_loss.backward()

                print("==> loss: %.4f\tmask_loss: %.2f\tpixel_loss: %.2f\trefine_loss: %.2f" 
                                % (
                                    total_loss.item(),
                                    mask_loss.item(),
                                    pixel_loss.item(),
                                    refine_loss.item()
                                    # stability_loss.item()
                                ))

                # No use
                if total_loss.item() > loss_threshold:
                    break

                # TODO: why .mean()? wm.grad should have the same size as wm
                sign = wm.grad.sign().mean()
                # TODO: change to gradient descent
                # minus -
                wm_adv = wm + alpha * sign * mask

                eta = torch.clamp(wm_adv - wm_ori, min=-eps, max=eps)
                wm = torch.clamp(wm + eta, min=min_val, max=1).detach_()

                del inputs
                del sign
            del images
        
        # validate for every epoch
        cur_wm = wm.clone().detach()
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
        # save if it is a better result
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

    # all epochs finish
    torch.save(mask, dir_path + "/wm_adv_mask_0.pt")
    if is_save:

        # TODO: it is redundant?
        wm_adv = torch.clamp(wm, min=min_val, max=1).detach_()
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

    return torch.clamp(wm, min=min_val, max=1).detach_()
