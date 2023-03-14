import torchvision.transforms.functional as TF
import argparse
import os
import sys
import torch
from options import Options
from src.utils.data_process import get_data_list
from src.models import bvmr, slbr, split
from src.utils.train_utils_adam_l2 import build_noise
from src.utils.watermark_gen import gen_wm
from src.utils.image_process import read_tensor, save_adv_pic, transform_pos, rotate_img_ts
torch.set_printoptions(profile="full")


def main():
    parser = Options(is_train=True).parser
    args = parser.parse_args()
    print("---------------------------args---------------------------")
    for k in list(vars(args).keys()):
        print("==> \033[1;35m%s\033[0m: %s" % (k, vars(args)[k]))
    print("---------------------------args---------------------------")
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # device = torch.device("cuda:0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.standard_transform:
        min_val, max_val = 0, 1
    else:
        min_val, max_val = -1, 1

    # Different models
    if args.model == 'bvmr':
        gt_model = bvmr(args=args, device=device)
    elif args.model == 'slbr':
        gt_model = slbr(args=args, device=device)
    elif args.model == 'split':
        gt_model = split(args=args, device=device)
    else:
        print("This model({}) doesn't support currently!".format(args.model))
        sys.exit(1)

    data_root = "./datasets/CLWD/train"
    wm_path = "%s/Watermark" % (data_root)
    mask_path = "%s/Mask" % (data_root)
    bg_path = "%s/Watermark_free_image" % (data_root)
    im_path = "%s/Watermarked_image" % (data_root)

    # data_lis = [{"wm", "mask", "bg", "im": str of path}, ...]
    data_lis = get_data_list(
        wm_path=wm_path,
        mask_path=mask_path,
        bg_path=bg_path,
        im_path=im_path,
        shuffle=True,
    )

    patch, mask = gen_wm(text=args.text, text_size=args.text_size,
                         standard_norm=args.standard_transform, device=device)
    print('patch:', patch.size())
    print('mask:', mask.size())

    eta = build_noise(
        model=gt_model,
        model_name=args.model,
        wm=patch,
        mask=mask,
        epoch=args.epoch,
        data_lis=data_lis,
        alpha=args.lr,
        eps=int(args.eps*255)/255.,
        batch_size=args.batch_size,
        sample_num=args.sample_num,  # times 10
        transparences=args.transparences,
        is_save=True,
        device=device,
        min_val=min_val,
        log_dir=args.log
    )

    # eta = eta[[2, 1, 0], :, :, ]
    N_imgs = 20
    bgs = [
        read_tensor(data_lis[250 + i], add_dim=False, standard_transform=args.standard_transform)[1].to(device) for i in range(N_imgs)
    ]

    transpar = torch.linspace(0.3, 0.7, N_imgs)
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
    masks = [wm_mask_affine[j][3:, :, :].repeat(
        3, 1, 1) for j in range(N_imgs)]
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

    
    save_adv_pic(
        path=args.output_dir + args.model + '/',
        image_lis=im_ts,
        mask_lis=list(zip(etas, masks)),
        standard_transform=args.standard_transform,
        bg_lis=bgs
    )

    pass


if __name__ == "__main__":
    main()
