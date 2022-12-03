import argparse
import os
import sys
import torch
import cv2
from options import Options
from src.utils.data_process import get_data_list
from src.models import bvmr, slbr, split
from src.utils.data_process import collect_synthesized
from src.utils.image_process import load_image, save_adv_pic, transform_pos, deploy_wm, translation_img_ts

torch.set_printoptions(profile="full")


def main():
    parser = Options(deploy=True).parser
    args = parser.parse_args()
    print("---------------------------args---------------------------")
    for k in list(vars(args).keys()):
        print("==> \033[1;35m%s\033[0m: %s" % (k, vars(args)[k]))
    print("---------------------------args---------------------------")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir[-1] != "/":
        args.output_dir += "/"

    bgs_path_lis = collect_synthesized(args.bgs_dir)
    bg_ts = torch.stack(
        [
            load_image(path=path, standard_norm=args.standard_transform)[1].to(device)
            for path in bgs_path_lis
        ],
        dim=0,
    )
    wm_ts = load_image(path=args.wm_path, standard_norm=args.standard_transform)[1].to(
        device
    )
    mask_ts = load_image(path=args.mask_path,gray=True)[1].to(device)
    mask_ts[mask_ts!=0] = 1
    for bg in bg_ts:
        new_im_ts, new_wm_ts, new_expanded_mask, new_mask  = deploy_wm(
            wm=wm_ts,
            mask=mask_ts,
            bg=bg,
            opacity=0.5,
            size_proportion=(0.7, 0.7),
            position=(10, 20),
            rotation_angle=30,
            standard_transform=args.standard_transform,
        )
        break

    save_adv_pic(
        path=args.output_dir,
        image_lis=new_im_ts.unsqueeze(0),
        mask_lis=[(new_wm_ts, new_expanded_mask)],
        bg_lis=[bg_ts[0]],
        standard_transform=args.standard_transform,
    )

    pass


if __name__ == "__main__":
    main()
