import random
import torch
import cv2
import os
import numpy as np
from .image_process import read_tensor
from tqdm import tqdm
import time

import torchvision.transforms as T
import torchvision.transforms.functional as TF

# total_loss = -1 * mask_loss + 1.e2 * pixel_loss + 1.e1 * refine_loss + 0.1 * stability_loss

# loss_a, loss_b, loss_c, loss_d, loss_e = -8e-1, 0.e2, 10.e1, 1.e-1, 0.e1
loss_a, loss_c, loss_e = 50, 5, 1e-2

def cal_refine_loss(reconstructed_images, inputs, masks, trans):
    # masks = masks.repeat(1,3,1,1)
    trans = trans.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return (
            torch.square((reconstructed_images - inputs) * masks).sum(dim=(1, 2, 3))
            / torch.sum(masks == 1, dim=(1, 2, 3), keepdim=True)
            / ((1 - trans) ** 2)
        ).mean()

def cal_mask_loss(guess_mask, masks, bce):
    # return 1/bce(guess_mask, masks)
    return torch.square(guess_mask).mean(dim=(1, 2, 3)).mean()

def cal_stab_loss(guess_images, noise_guess_images, noise):
    sigma = 0.95
    norm_dy = torch.sum(torch.abs(guess_images-noise_guess_images),
                    dim=[1,2,3])

    norm_dx = torch.sum(torch.abs(noise),
                    dim=[1,2,3])
    return torch.nn.functional.relu(norm_dy-sigma*norm_dx).mean()


def process_model_output(outputs, model_name):
    if model_name == "bvmr":
        return outputs[0], outputs[1]
    elif model_name == "slbr":
        return outputs[0][0], outputs[1][0]
    elif model_name == "split":
        return outputs[0][0], outputs[1]


def valid_noise(
    model, 
    model_name, 
    wm, 
    wm_ori, 
    mask, 
    valid_data_lis, 
    device, 
    EOT, 
    EOT_color, 
    min_val,
    batch_size
):  
    with torch.no_grad():
        bce = torch.nn.BCELoss(reduction="mean")
        model.model.eval()

        # trans_eot = transparences * len(valid_data_lis)
        # valid_data_eot = valid_data_lis * len(transparences)
        sample_num = len(valid_data_lis)
        all_loss = []

        # transpar = torch.linspace(0.3, 0.7, sample_num)
        angles = torch.linspace(-45, 45, sample_num).numpy()
        dx = torch.linspace(-int(256*0.2), int(256*0.2), sample_num).numpy()
        dy = torch.linspace(-int(256*0.3), int(256*0.3), sample_num).numpy()

        for j in range(sample_num // batch_size):
            bgs = [
                read_tensor(
                    valid_data_lis[i + j * batch_size],
                    standard_transform=(min_val != -1),
                    add_dim=True,
                )[1].to(device)
                for i in range(batch_size)
            ]
            batch_angles = angles[j*batch_size: (j+1)*batch_size]
            batch_dx = dx[j*batch_size: (j+1)*batch_size]
            batch_dy = dy[j*batch_size: (j+1)*batch_size]

            wm_mask = (
                torch.cat((wm, mask), dim=0)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
            ).to(device)
            # the first stage EOT, transform wm and mask (spatial transforms)
            # wm_mask = [EOT(wm_mask[i : i + 1]) for i in range(wm_mask.size(0))]
            wm_mask = [TF.affine(wm_mask[i : i + 1], 
                            angle=int(batch_angles[i]), 
                            translate=(batch_dx[i], batch_dy[i]),
                            scale=1,
                            shear=0) 
                            for i in range(batch_size)]
            # the second stage EOT, only transform wm (color adjustments)
            # wm_eot = [EOT_color(_wm_mask[:, :3, :, :]) for _wm_mask in wm_mask]
            wm_eot = [_wm_mask[:, :3, :, :] for _wm_mask in wm_mask]
            mask_eot = [
                _wm_mask[:, 3:, :, :] for _wm_mask in wm_mask
            ]  # do not transform mask
            # transparence from 0.3 to 0.7
            trans_eot = torch.linspace(0.3, 0.7, batch_size).to(device)

            images = []
            for i in range(batch_size):
                images.append(bgs[i] * ((1 - mask_eot[i]) + mask_eot[i] * trans_eot[i])
                                + wm_eot[i] * mask_eot[i] * (1 - trans_eot[i]))

            inputs = torch.vstack(images)
            mask_eot = torch.vstack(mask_eot)

            # print("before", inputs.size())
            # noise = (torch.randn(inputs.size())*0.01).to(device)
            # inputs = torch.vstack([inputs, inputs+noise])
            # print("after", inputs.size())

            # =====================================
            if model_name == 'bvmr':
                outputs = model.model(inputs*2-1)
            else:
                outputs = model.model(inputs)
            guess_images, guess_mask = process_model_output(outputs, model_name)
            
            if model_name == 'bvmr':
                guess_images = (guess_images+1)/2
            # =====================================
            # noise_guess_images = guess_images[batch_size:]
            # guess_images = guess_images[:batch_size]
            # guess_mask = guess_mask[:batch_size]
            # inputs = inputs[:batch_size]

            # stability_loss = cal_stab_loss(
            #         guess_images, 
            #         noise_guess_images,
            #         noise
            #     )
            # expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
            expanded_guess_mask = mask_eot.repeat(1, 3, 1, 1)
            reconstructed_pixels = guess_images * expanded_guess_mask
            reconstructed_images = inputs * (1 - expanded_guess_mask) + reconstructed_pixels

            mask_loss = cal_mask_loss(guess_mask, mask_eot, bce)
            # pixel_loss = torch.abs(wm - wm_ori).mean()
            # refine_loss = torch.abs(reconstructed_images - inputs).mean()
            refine_loss = cal_refine_loss(
                                        reconstructed_images, 
                                        inputs, 
                                        mask_eot, 
                                        trans_eot
                                    )
            
            total_loss = (
                # loss_a * mask_loss + loss_b * pixel_loss + loss_c * refine_loss
                loss_a * mask_loss + loss_c * refine_loss #+ loss_e * stability_loss

            )
            all_loss.append(total_loss.item())

    return sum(all_loss)/len(all_loss)


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
    valid_size=50,
    max_iters=2,
    alpha=1 / 255,
    eps=0.01,
    loss_threshold=-5,
    transparences=[0.3, 0.5, 0.7],
    is_save=False,
    log_dir=None
):
    eta = None
    tmp_loss = 100.0

    dir_path, file_name = "/home/public/imprints/exp_data/" + model_name, "/wm_adv_0.pt"

    bce = torch.nn.BCELoss(reduction="mean")
    model.model.eval()

    # validation host images
    print('sample_num:',sample_num ,'validation size:', valid_size)
    valid_data_lis = data_lis[sample_num : sample_num + valid_size]
    # training host images
    data_lis = data_lis[:sample_num]

    # EOT
    EOT = T.Compose(
        [
            T.RandomAffine(
               degrees=(-90, 90), translate=(0.4, 0.4), scale=(0.5, 1.5), fill=min_val
            ),
            T.RandomPerspective(distortion_scale=0.1, p=0.5, fill=min_val),
        ]
    )
    EOT_color = T.Compose(
        [
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ]
    )


    # initial watermark
    wm_ori = wm.clone().detach()

    wm.requires_grad = True
    optimizer = torch.optim.Adam([wm], lr=0.01)

    for epoch_no in range(epoch):
        random.shuffle(data_lis)

        for j in range(sample_num // batch_size):
            bgs = [     # [1, 3, 256, 256]
                read_tensor(
                    data_lis[i + j * batch_size],
                    standard_transform=(min_val != -1),
                    add_dim=True,
                )[1].to(device)
                for i in range(batch_size)
            ]

            for _ in range(max_iters):
                # requires_grad
                
                optimizer.zero_grad()
                model.model.zero_grad()
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
                trans_eot = torch.FloatTensor([torch.rand(1) * 0.8 + 0.1 
                                            for _ in range(len(wm_eot))]).to(device)

                images = []
                for i in range(batch_size):
                  images.append(bgs[i] * ((1 - mask_eot[i]) + mask_eot[i] * trans_eot[i])
                                    + wm_eot[i] * mask_eot[i] * (1 - trans_eot[i]))

                inputs = torch.vstack(images)
                mask_eot = torch.vstack(mask_eot)

            # =====================================
                if model_name == 'bvmr':
                    outputs = model.model(inputs*2-1)
                else:
                    outputs = model.model(inputs)
                # guess_images: [batch, 3, 256, 256]
                guess_images, guess_mask = process_model_output(outputs, model_name)
                
                if model_name == 'bvmr':
                    guess_images = (guess_images+1)/2
            # =====================================

                expanded_guess_mask = mask_eot #.repeat(1, 3, 1, 1)

                reconstructed_pixels = guess_images * expanded_guess_mask
                reconstructed_images = (
                    inputs * (1 - expanded_guess_mask) + reconstructed_pixels
                )


                mask_loss = cal_mask_loss(guess_mask, mask_eot, bce)

                refine_loss = cal_refine_loss(
                                    reconstructed_images, 
                                    inputs, 
                                    mask_eot, 
                                    trans_eot
                                )


                if epoch_no >= epoch//2:
                    total_loss = loss_a * mask_loss + loss_c * refine_loss #+ loss_e * stability_loss
                else:
                    total_loss = loss_c * refine_loss #+ loss_e * stability_loss
                    mask_loss = torch.FloatTensor([0])
                
                total_loss.backward(retain_graph=True)

                print(
                   "==> loss↓: %.4f\tmask_loss↑: %.2f(%.2f)\trefine_loss↓: %.4f(%.4f)" #\tstab↓:%.4f(%.4f)"
                    % (
                        total_loss.item(),
                        mask_loss.item(), mask_loss.item()*loss_a,
                        refine_loss.item(), refine_loss.item()*loss_c,

                    )
                )

                # No use
                if total_loss.item() < loss_threshold:
                    break

                wm.grad = wm.grad * mask
                optimizer.step()

                with torch.no_grad():
                    eta = torch.clamp(wm - wm_ori, min=-eps, max=eps)
                    wm[:] = torch.clamp(wm_ori + eta, min=min_val, max=1) # .detach_()

                del inputs
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
            device=device,
            EOT=EOT,
            EOT_color=EOT_color,
            min_val=min_val,
            batch_size=batch_size
        )
        # save if it is a better result, lower loss is better
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        # save the latest version   
        torch.save(cur_wm, dir_path + file_name)
        torch.save(cur_wm, log_dir + file_name)

        # save the lowest loss version
        if cur_loss < tmp_loss:
            tmp_loss = cur_loss
            best_wm = cur_wm.clone()
            torch.save(best_wm, "wm_adv_0.pt")
            torch.save(mask, "wm_adv_mask_0.pt")

            if min_val == -1:
                best_wm_tmp = (best_wm) / 2 + 0.5
            else:
                best_wm_tmp = best_wm / 1
            best_wm_tmp = best_wm_tmp.cpu().numpy()
            best_wm_tmp = np.transpose(best_wm_tmp, (1, 2, 0)) * 255
            best_wm_tmp.astype(np.uint8)

            cv2.imwrite(dir_path + "/wm_adv_best.png", best_wm_tmp[:, :, [2,1,0]])
            cv2.imwrite(log_dir + "/wm_adv_best.png", best_wm_tmp[:, :, [2,1,0]])


        if min_val == -1:
            cur_wm = (cur_wm) / 2 + 0.5
        cur_wm = cur_wm.cpu().numpy()
        cur_wm = np.transpose(cur_wm, (1, 2, 0)) * 255
        cur_wm.astype(np.uint8)
        cv2.imwrite(dir_path + "/wm_adv_latest.png", cur_wm[:, :, [2,1,0]])
        cv2.imwrite(log_dir + "/wm_adv_latest.png", cur_wm[:, :, [2,1,0]])          

        print(
            "==> epoch_no: {}  currrent_loss: {:.4f}   best_loss: {:.4f}".format(
                epoch_no, cur_loss, tmp_loss
            )
        )

    # all epochs finish
    torch.save(mask, dir_path + "/wm_adv_mask_end.pt")
    torch.save(mask, log_dir + "/wm_adv_mask_end.pt")


    return torch.clamp(best_wm, min=min_val, max=1).detach_()


