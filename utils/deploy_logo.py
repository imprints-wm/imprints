import argparse
import os
import sys
import torch
from options import Options
from src.utils.data_process import get_data_list
from src.models import bvmr,slbr,split
from src.utils.train_utils_adam_l2 import build_noise
from src.utils.watermark_gen import gen_wm, load_logo
from src.utils.image_process import read_tensor, save_adv_pic, transform_pos, rotate_img_ts
torch.set_printoptions(profile="full")

import torchvision.transforms.functional as TF

def main():
    parser = Options(is_train=True).parser
    args = parser.parse_args()
    print("---------------------------args---------------------------")
    for k in list(vars(args).keys()):
        print("==> \033[1;35m%s\033[0m: %s" % (k, vars(args)[k]))
    print("---------------------------args---------------------------")
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.standard_transform:
        min_val, max_val = 0, 1
    else:
        min_val, max_val = -1, 1

    data_root = "./datasets/CLWD/test"
    wm_path = "%s/Watermark_free_image" % (data_root)
    mask_path = "%s/Mask" % (data_root)
    bg_path = "%s/Watermark_free_image" % (data_root)
    im_path = "%s/Watermarked_image" % (data_root)


    data_lis = get_data_list(
        wm_path=wm_path,
        mask_path=mask_path,
        bg_path=bg_path,
        im_path=im_path,
        shuffle=False,
    )


    eta = torch.load(args.logo_path+"/wm_adv_0.pt")
    mask = torch.load(args.logo_path+"/wm_adv_mask_end.pt")

    N_imgs = 20
    bgs = [
        read_tensor(data_lis[i], add_dim=False, standard_transform=args.standard_transform)[1].to(device) for i in range(N_imgs)
    ]
    transpar = torch.linspace(0.45, 0.75, N_imgs)
    angles = torch.linspace(-45, 45, N_imgs).numpy()
    dx = torch.linspace(-int(256*0.2), int(256*0.2), N_imgs).numpy()
    dy = torch.linspace(-int(256*0.3), int(256*0.3), N_imgs).numpy()

    wm_mask = torch.vstack([eta, mask])
    wm_mask_affine = [TF.affine(wm_mask, 
                            angle=int(angles[j]), 
                            translate=(dx[j], dy[j]),
                            scale=1,
                            shear=0) 
                            for j in range(N_imgs)]
    

    etas = [wm_mask_affine[j][:3, :, :] for j in range(N_imgs)]
    masks = [wm_mask_affine[j][3:, :, :].repeat(3,1,1) for j in range(N_imgs)]

    im_ts = torch.stack(
        [
            torch.clamp(
                bgs[j] * (1 - masks[j])
                + bgs[j] * masks[j] * transpar[j]
                + etas[j] * masks[j] * (1 - transpar[j]),
                min=min_val,
                max=max_val,
            ).detach_()
            for j in range(N_imgs)
        ],
        dim=0,
    )
    print(args.output_dir)
    save_adv_pic(
        path=args.output_dir, 
        image_lis=im_ts, 
        mask_lis=list(zip(etas, masks)), 
        standard_transform=args.standard_transform,
        bg_lis=bgs
    )

    pass


if __name__ == "__main__":
    main()
