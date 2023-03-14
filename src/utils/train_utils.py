import random
import torch
import cv2
import os
import numpy as np
from .image_process import read_tensor
from tqdm import tqdm
import time

import torchvision.transforms as T
from torch.autograd.functional import jacobian

# total_loss = -1 * mask_loss + 1.e2 * pixel_loss + 1.e1 * refine_loss + 0.1 * stability_loss

loss_a, loss_b, loss_c, loss_d, loss_e = -8e-1, 0.e2, 3.e1, 1.e-1, 5.e1

def process_model_output(outputs, model_name):
    if model_name == "bvmr":
        return outputs[0], outputs[1]
    elif model_name == "slbr":
        return outputs[0][0], outputs[1][0]
    elif model_name == "split":
        return outputs[0][0], outputs[1]


def valid_noise(
    model, model_name, wm, wm_ori, mask, valid_data_lis, transparences, min_val, device
):  
    with torch.no_grad():
        bce = torch.nn.BCELoss(reduction="mean")
        mse = torch.nn.MSELoss(reduction="mean")
        model.model.eval()

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
        # inputs.requires_grad = False

        outputs = model.model(inputs)
        guess_images, guess_mask = process_model_output(outputs, model_name)
        model.model.zero_grad()

        expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
        reconstructed_pixels = guess_images * expanded_guess_mask
        reconstructed_images = inputs * (1 - expanded_guess_mask) + reconstructed_pixels

        mask_loss = bce(guess_mask, masks)
        pixel_loss = torch.abs(wm - wm_ori).mean()
        refine_loss = (
                    torch.abs((reconstructed_images - inputs) * masks)
                    / torch.sum(masks == 1, dim=(1, 2, 3), keepdim=True)
                ).mean()
        
        total_loss = (
            loss_a * mask_loss + loss_b * pixel_loss + loss_c * refine_loss
        )
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
    alpha=1 / 255,
    eps=0.01,
    loss_threshold=-5,
    transparences=[0.3, 0.5, 0.7],
    is_save=False,
    log_dir=None
):
    eta = None
    tmp_loss = 5.0

    dir_path, file_name = "./ckpt_wm/" + model_name, "/wm_adv_0.pt"

    bce = torch.nn.BCELoss(reduction="mean")
    mse = torch.nn.MSELoss(reduction="mean")
    model.model.eval()

    # validation host images
    valid_data_lis = data_lis[sample_num : sample_num + valid_size]
    # training host images
    data_lis = data_lis[:sample_num]

    # initial watermark
    wm_ori = wm.clone().detach()

    # EOT
    EOT = T.Compose(
        [
            T.RandomAffine(
                degrees=(-45, 45), translate=(0.1, 0.3), scale=(0.2, 1), fill=min_val
            ),
            T.RandomPerspective(distortion_scale=0.2, p=1.0, fill=min_val),
        ]
    )
    EOT_color = T.Compose(
        [
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 5))
        ]
    )

    for epoch_no in range(epoch):
        random.shuffle(data_lis)
        # save and load the watermark every epoch
        if os.path.exists(dir_path + file_name): #and epoch_no != 0:
            wm = torch.load(dir_path + file_name, map_location=device)

        for j in range(sample_num // batch_size):
            # a batch of host images, batchsize * transparences
            bgs = [ # [1, 3, 256, 256]
                read_tensor(
                    data_lis[i + j * batch_size],
                    standard_transform=(min_val != -1),
                    add_dim=True,
                )[1].to(device)
                for i in range(batch_size)
            ]

            for k in range(max_iters):
                # requires_grad
                wm.requires_grad = True

                wm_mask = (
                    torch.cat((wm, mask), dim=0)
                    .unsqueeze(0)
                    .repeat(batch_size, 1, 1, 1)
                ).to(device)
                # the first stage EOT, transform wm and mask (spatial transforms)
                wm_mask = [EOT(wm_mask[i : i + 1]) for i in range(wm_mask.size(0))]
                # the second stage EOT, only transform wm (color adjustments)
                wm_eot = [EOT_color(_wm_mask[:, :3, :, :]) for _wm_mask in wm_mask]
                mask_eot = [
                    _wm_mask[:, 3:, :, :] for _wm_mask in wm_mask
                ]  # do not transform mask
                trans_eot = [torch.rand(1).to(device) * 0.6 + 0.2 for _ in range(len(wm_eot))]

                images = []
                for i in range(batch_size):
                    images.append(bgs[i] * ((1 - mask_eot[i]) + mask_eot[i] * trans_eot[i])
                                    + wm_eot[i] * mask_eot[i] * (1 - trans_eot[i]))

                inputs = torch.vstack(images)
                mask_eot = torch.vstack(mask_eot)

                outputs = model.model(inputs)
                # guess_images: [batch, 3, 256, 256]
                guess_images, guess_mask = process_model_output(outputs, model_name)
                model.model.zero_grad()
                expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
                reconstructed_pixels = guess_images * expanded_guess_mask
                reconstructed_images = (
                    inputs * (1 - expanded_guess_mask) + reconstructed_pixels
                )

                # guess_mask, mask_eot: [batch, 1, 256, 256]
                area = ((guess_mask==1)+(mask_eot==1))>=1
                mask_loss = bce(guess_mask[area], 
                                mask_eot[area])
                pixel_loss = torch.abs(wm - wm_ori).mean()
                consistent_loss = (
                                wm[0:1,:,:][mask==1].std()
                                + wm[1:2,:,:][mask==1].std()
                                + wm[2:3,:,:][mask==1].std()
                                )/3

                refine_loss = (
                    torch.abs((reconstructed_images - inputs) * mask_eot).sum(dim=(1, 2, 3))
                    / torch.sum(mask_eot == 1, dim=(1, 2, 3))
                ).mean()

                noise = (torch.randn(inputs.size())*1/255).to(device)
                est_grad = (model.endtoend_func(inputs+noise)-model.endtoend_func(inputs-noise))\
                                    /(2*noise+1e-6)
                stability_loss = torch.abs(est_grad).mean()

                
                total_loss = loss_a * mask_loss + loss_b * pixel_loss + loss_c * refine_loss + loss_d * stability_loss + loss_e * consistent_loss
                total_loss.backward()

                print(
                    "==> loss↓: %.4f\tmask_loss↑: %.2f(%.2f)\tpixel_loss↓: %.4f(%.4f)\trefine_loss↓: %.4f(%.4f)\tstability↓:%.4f(%.4f)\tconsistent↓:%.4f(%.4f)"
                    % (
                        total_loss.item(),
                        mask_loss.item(), mask_loss.item()*loss_a,
                        pixel_loss.item(), pixel_loss.item()*loss_b,
                        refine_loss.item(), refine_loss.item()*loss_c,
                        stability_loss.item(), stability_loss.item()*loss_d,
                        consistent_loss.item(), consistent_loss.item()*loss_e
                    )
                )

                # No use
                if total_loss.item() < loss_threshold:
                    break

                sign = wm.grad.sign()  # .mean()
                wm_adv = wm - alpha * sign * mask

                eta = torch.clamp(wm_adv - wm_ori, min=-eps, max=eps)
                wm = torch.clamp(wm_ori + eta, min=min_val, max=1).detach_()

                del inputs
                del sign
            del images

        # validate for every epoch
        cur_wm = wm.clone().detach()
        cur_loss = valid_noise(
            model=model,
            model_name=model_name,
            wm=cur_wm,
            wm_ori=wm_ori,
            mask=mask,
            valid_data_lis=valid_data_lis,
            transparences=transparences,
            min_val=min_val,
            device=device,
        )

        # save if it is a better result, lower loss is better
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        torch.save(cur_wm, dir_path + file_name)
        torch.save(cur_wm, log_dir + file_name)

        if min_val == -1:
            cur_wm = (cur_wm) / 2 + 0.5
        cur_wm = cur_wm.cpu().numpy()
        cur_wm = np.transpose(cur_wm, (1, 2, 0)) * 255
        cur_wm.astype(np.uint8)
        cv2.imwrite(dir_path + "/wm_adv_latest.png", cur_wm)
        cv2.imwrite(log_dir + "/wm_adv_latest.png", cur_wm)

        if cur_loss < tmp_loss:
            tmp_loss = cur_loss
            cv2.imwrite(dir_path + "/wm_adv_best.png", cur_wm)
            cv2.imwrite(log_dir + "/wm_adv_best.png", cur_wm)

        print(
            "==> epoch_no: {}  currrent_loss: {:.4f}   best_loss: {:.4f}".format(
                epoch_no, cur_loss, tmp_loss
            )
        )


    # all epochs finish
    torch.save(mask, dir_path + "/wm_adv_mask_end.pt")
    torch.save(mask, log_dir + "/wm_adv_mask_end.pt")

    if is_save:

        wm_adv = torch.clamp(wm, min=min_val, max=1).detach_()
        torch.save(wm_adv, "wm_adv_0.pt")
        torch.save(mask, "wm_adv_mask_0.pt")
        if min_val == -1:
            wm_adv = (wm_adv) / 2 + 0.5
        wm_adv = wm_adv.cpu().numpy()
        wm_adv = np.transpose(wm_adv, (1, 2, 0)) * 255
        wm_adv.astype(np.uint8)
        cv2.imwrite("wm_adv_0.png", wm_adv)

    return torch.clamp(wm, min=min_val, max=1).detach_()
