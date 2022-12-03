import argparse
import os
import sys
import torch
from options import Options
from src.utils.data_process import get_data_list
from src.models import bvmr, slbr, split
from src.utils.data_process import collect_synthesized
from src.utils.image_process import load_image, save_adv_pic, transform_pos
from tqdm import tqdm

torch.set_printoptions(profile="full")


def main():
    parser = Options(is_train=False).parser
    args = parser.parse_args()
    print("---------------------------args---------------------------")
    for k in list(vars(args).keys()):
        print("==> \033[1;35m%s\033[0m: %s" % (k, vars(args)[k]))
    print("---------------------------args---------------------------")
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "bvmr":
        gt_model = bvmr(args=args, device=device)
    elif args.model == "slbr":
        gt_model = slbr(args=args, device=device)
    elif args.model == "split":
        gt_model = split(args=args, device=device)
    else:
        print("This model({}) doesn't support currently!".format(args.model))
        sys.exit(1)

    if args.standard_transform:
        min_val, max_val = 0, 1
    else:
        min_val, max_val = -1, 1
    
    if args.output_dir[-1] != '/':
        args.output_dir += '/'

    N_img = 20
    img_path_lis = collect_synthesized(args.input_dir)[:20]
    bgs_path_lis = collect_synthesized(args.bgs_dir)[:20]
    im_ts = torch.stack(
        [
            load_image(path=path, standard_norm=args.standard_transform)[1].to(device)
            for path in img_path_lis
        ],
        dim=0,
    )
    bg_ts = torch.stack(
        [
            load_image(path=path, standard_norm=args.standard_transform)[1].to(device)
            for path in bgs_path_lis
        ],
        dim=0,
    )
    # print(im_ts.shape)

    gt_model.model.eval()

    for i in tqdm(range(args.iteration)):
        with torch.no_grad():
            outputs = gt_model.model(im_ts)
            guess_images, guess_masks = gt_model.resolve(outputs)
            # print(guess_images.shape)
            # print(guess_masks.shape)
            # print(outputs[2])

            expanded_guess_mask = guess_masks.repeat(1, 3, 1, 1)
            reconstructed_pixels = guess_images * expanded_guess_mask
            reconstructed_images = im_ts * (1 - expanded_guess_mask) + reconstructed_pixels
            im_ts = reconstructed_images
    

    if outputs[2][0] is not None:
        guess_wms = outputs[2] * expanded_guess_mask
    else:
        guess_wms = im_ts - reconstructed_images

    if args.standard_transform == 0:
        guess_wms = (guess_wms - 1) * expanded_guess_mask + 1

    mask_lis = [(a, b) for a, b in zip(guess_wms, expanded_guess_mask)]

    save_adv_pic(
        path=args.output_dir + args.model + '/',
        image_lis=reconstructed_images,
        bg_lis=bg_ts,
        mask_lis=mask_lis,
        standard_transform=args.standard_transform,
        filename_list=img_path_lis,
        train=False,
    )


if __name__ == "__main__":
    main()
