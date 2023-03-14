import torchvision.transforms.functional as TF
import argparse
import os
import sys
import torch
from options import Options
from src.utils.data_process import get_data_list
from src.models import bvmr, slbr, split
from src.utils.train_utils_adam_l2 import build_noise
from src.utils.watermark_gen import gen_wm, load_logo
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

    data_root = "./datasets/CLWD/test"
    # data_root = "/home/public/imprints/exp_data/vaccine0"
    print("Adding wms to", data_root)
    # wm_path = "%s/Watermark_free_image" % (data_root)
    # mask_path = "%s/Mask" % (data_root)
    # bg_path = "%s/Watermark_free_image" % (data_root)
    # im_path = "%s/Watermarked_image" % (data_root)
    wm_path = "%s/Watermark_free_image" % (data_root)
    mask_path = "%s/Watermark_free_image" % (data_root)
    bg_path = "%s/Watermark_free_image" % (data_root)
    im_path = "%s/Watermark_free_image" % (data_root)

    # data_lis = [{"wm", "mask", "bg", "im": str of path}, ...]
    data_lis = get_data_list(
        wm_path=wm_path,
        mask_path=mask_path,
        bg_path=bg_path,
        im_path=im_path,
        shuffle=False,
    )

    change_pos = True if args.change_pos.lower() == "true" else False
    change_siz = True if args.change_siz.lower() == "true" else False
    change_ang = True if args.change_ang.lower() == "true" else False
    change_opa = True if args.change_opa.lower() == "true" else False
    add_noise = True if args.add_noise.lower() == "true" else False
    random_opa = True if args.random_opa.lower() == "true" else False

    print('change_pos', change_pos)
    print('change_siz', change_siz)
    print('change_ang', change_ang)
    print('change_opa', change_opa)
    print('add_noise', add_noise)
    print('random_opa', random_opa)

    # the order of the optimized watermarks
    if os.path.exists(args.logo_path+"/"+os.listdir(args.logo_path)[0]+"/wm_adv_0.pt"):
        logo_dirs = os.listdir(args.logo_path)
        logo_dirs.sort()
    else:
        # the order of the original watermarks should be changed
        logo_dirs = list(range(10, 20)) + [1] \
            + list(range(20, 30)) + [2] \
            + list(range(30, 40)) + [3] \
            + [40] + list(range(4, 10))
        logo_dirs = [
            str(idx)+'.png' for idx in logo_dirs][:len(os.listdir(args.logo_path))]

    for idx, logo_dir in enumerate(logo_dirs):
        if os.path.exists(args.logo_path+"/"+logo_dir+"/wm_adv_0.pt"):
            eta = torch.load(args.logo_path+"/"+logo_dir+"/wm_adv_0.pt")
            mask = torch.load(args.logo_path+"/"+logo_dir +
                              "/wm_adv_mask_end.pt")
        else:
            eta, mask = load_logo(logo_path=args.logo_path+"/"+logo_dir,
                                  standard_norm=args.standard_transform,
                                  device=device)
        print('==> patch:', eta.size())
        print('==> mask:', mask.size())

        if add_noise:
            noise = torch.clamp(
                (torch.randn(eta.size())*0.05).to(device), min=-0.2, max=0.2)
            eta = torch.clamp(eta+noise, min=0, max=1)

            print(
                "noise: {:.3f}-{:.3f}".format(noise.max().item(), noise.min().item()))
            print(
                "eta: {:.3f}-{:.3f}".format(eta.max().item(), eta.min().item()))
            eta = torch.clamp(eta + noise, min=0, max=1)
        else:
            pass

        # eta = eta[[2, 1, 0], :, :, ]
        N_imgs = 10000//40
        bgs = [
            read_tensor(data_lis[i+idx*N_imgs], add_dim=False, standard_transform=args.standard_transform)[1].to(device) for i in range(N_imgs)
        ]
        if change_opa:
            opa_start = 0.4
            opa_end = 0.8
            trans_start = 1-opa_end
            trans_end = 1-opa_start
            print(f"opacity: {opa_start}-{opa_end}",
                  f"trans: {trans_start}-{trans_end}")
            transpar = torch.linspace(trans_start, trans_end, N_imgs)
        else:
            opa = 1-args.opa_value
            print("opacity", args.opa_value, "transpar", opa)
            transpar = torch.linspace(opa, opa, N_imgs)

        if random_opa:
            opa = 1-args.opa_value
            print("opacity", args.opa_value, "transpar", opa)
            variation = 0.1
            print("opacity variation", variation)
            width = mask.size(1)
            seg = width//10
            transpar = []
            for _ in range(N_imgs):
                delta_opa = torch.rand([1, seg+1])*2*variation-variation
                delta_opa = delta_opa.repeat_interleave(width//seg)[:width]
                trans = torch.ones(mask.size())*opa
                trans += delta_opa
                transpar.append(trans.to(device))

        if change_ang:
            angles = torch.linspace(-45, 45, N_imgs).numpy()
        else:
            angles = torch.linspace(0, 0, N_imgs).numpy()

        if change_pos:
            dx = torch.linspace(-int(256*0.2), int(256*0.2), N_imgs).numpy()
            dy = torch.linspace(-int(256*0.3), int(256*0.3), N_imgs).numpy()
        else:
            dx = torch.linspace(-int(0), int(0), N_imgs).numpy()
            dy = torch.linspace(-int(0), int(0), N_imgs).numpy()

        if change_siz:
            sizes = torch.linspace(0.8, 1.2, N_imgs).numpy()
        else:
            # sizes = torch.linspace(1, 1, N_imgs).numpy()
            sizes = torch.linspace(
                args.size_value, args.size_value, N_imgs).numpy()



        wm_mask = torch.vstack([eta, mask])
        wm_mask_affine = [TF.affine(wm_mask,
                                    angle=int(angles[j]),
                                    translate=(dx[j], dy[j]),
                                    scale=sizes[j],
                                    shear=0)
                          for j in range(N_imgs)]


        etas = [wm_mask_affine[j][:3, :, :] for j in range(N_imgs)]
        masks = [wm_mask_affine[j][3:, :, :].repeat(
            3, 1, 1) for j in range(N_imgs)]
        imgs = [torch.clamp(
            bgs[j] * (1 - masks[j])
            + bgs[j] * masks[j] * transpar[j]
            + etas[j] * masks[j] * (1 - transpar[j]),
            min=min_val,
            max=max_val,
        ).detach_()
            for j in range(N_imgs)
        ]
        im_ts = torch.stack(imgs, dim=0)


        print(args.output_dir)
        save_adv_pic(
            path=args.output_dir,
            image_lis=im_ts,
            mask_lis=list(zip(etas, masks)),
            standard_transform=args.standard_transform,
            bg_lis=bgs,
            filename_list=[str(i+idx*N_imgs)+'.png' for i in range(N_imgs)]
        )


if __name__ == "__main__":
    main()
