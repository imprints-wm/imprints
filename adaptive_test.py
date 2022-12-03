import argparse
import os
import sys
import torch
from options import Options
from src.utils.data_process import get_data_list
from src.models import bvmr, slbr, split
from src.utils.data_process import collect_synthesized
from src.utils.image_process import load_image, save_adv_pic, transform_pos
import math
import random
import torchvision.transforms.functional as TF

torch.set_printoptions(profile="full")
random.seed(10086)

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

    batch_size=100
    all_img_path_lis = collect_synthesized(args.input_dir)
    all_bgs_path_lis = collect_synthesized(args.bgs_dir)

    for batch_idx in range(len(all_img_path_lis)//batch_size+1):
        img_path_lis = all_img_path_lis[batch_idx*batch_size:(batch_idx+1)*batch_size]
        bgs_path_lis = all_bgs_path_lis[batch_idx*batch_size:(batch_idx+1)*batch_size]
        # filename_lis = [str(i)+'.png' 
        #                 for i in range(batch_idx*batch_size,(batch_idx+1)*batch_size)]

        # im_ts = torch.stack(
        #     [
        #         load_image(path=path, standard_norm=args.standard_transform)[1].to(device)
        #         for path in img_path_lis
        #     ],
        #     dim=0,
        # )
        im_ts = [
                load_image(path=path, standard_norm=args.standard_transform)[1].to(device)
                for path in img_path_lis
                ]
        bg_ts = torch.stack(
            [
                load_image(path=path, standard_norm=args.standard_transform)[1].to(device)
                for path in bgs_path_lis
            ],
            dim=0,
        )


        adaptive = "saturation" # "color"
        print("adaptive", adaptive)
        if adaptive == "affine":
            # print(im_ts.shape)
            random_angles = [random.randint(-45, 46) for _ in range(batch_size)]
            inverse_scale = [(abs(math.cos(ang*math.pi/180.0))
                                + abs(math.sin(ang*math.pi/180.0)))
                                for ang in random_angles]

            im_ts = torch.stack(
                    [TF.affine(im_ts[j], 
                            angle=0,
                            translate=(0,0),
                            scale=1.0/inverse_scale[j],
                            shear=0) 
                    for j in range(batch_size)
                    ],
                    dim=0,
            )
            im_ts = torch.stack(
                    [TF.affine(im_ts[j], 
                            angle=int(random_angles[j]),
                            translate=(0,0),
                            scale=1,
                            shear=0) 
                    for j in range(batch_size)
                    ],
                    dim=0,
            )
        elif adaptive == "brightness":
            random_adjust = [random.random()+0.5
                            for _ in range(batch_size)]
            im_ts = torch.stack(
                    [TF.adjust_brightness(im_ts[j], 
                            brightness_factor=random_adjust[j]) 
                    for j in range(batch_size)
                    ],
                    dim=0,
            )
        elif adaptive == "hue":
            random_adjust = [random.random()-0.5
                            for _ in range(batch_size)]
            im_ts = torch.stack(
                    [TF.adjust_hue(im_ts[j], 
                            hue_factor=random_adjust[j]) 
                    for j in range(batch_size)
                    ],
                    dim=0,
            ) 
        elif adaptive == "saturation":
            random_adjust = [random.random()+0.5
                            for _ in range(batch_size)]
            im_ts = torch.stack(
                    [TF.adjust_saturation(im_ts[j], 
                            saturation_factor=random_adjust[j]) 
                    for j in range(batch_size)
                    ],
                    dim=0,
            )               
            
        gt_model.model.eval()
        with torch.no_grad():
            outputs = gt_model.model(im_ts)
            guess_images, guess_masks = gt_model.resolve(outputs)
            print(guess_images.shape)
            print(guess_masks.shape)
            # print(outputs[2])

            expanded_guess_mask = guess_masks.repeat(1, 3, 1, 1)
            reconstructed_pixels = guess_images * expanded_guess_mask
            reconstructed_images = im_ts * (1 - expanded_guess_mask) + reconstructed_pixels
            
        if adaptive=="affine":
            reconstructed_images = torch.stack(
                    [TF.affine(reconstructed_images[j], 
                                angle = -int(random_angles[j]),
                                translate=(0,0),
                                scale = inverse_scale[j],
                                shear=0) 
                    for j in range(batch_size)
                    ],
                    dim=0,
            )
        elif adaptive=="brightness":
            reconstructed_images = torch.stack(
                    [TF.adjust_brightness(reconstructed_images[j], 
                            brightness_factor=1.0/random_adjust[j]) 
                    for j in range(batch_size)
                    ],
                    dim=0,
            )     

        elif adaptive=="hue":
            reconstructed_images = torch.stack(
                    [TF.adjust_hue(reconstructed_images[j], 
                            hue_factor=-random_adjust[j]) 
                    for j in range(batch_size)
                    ],
                    dim=0,
            )     
        elif adaptive=="saturation":
            reconstructed_images = torch.stack(
                    [TF.adjust_saturation(reconstructed_images[j], 
                            saturation_factor=1.0/random_adjust[j]) 
                    for j in range(batch_size)
                    ],
                    dim=0,
            )        

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
            filename_list=[i.replace("jpg", "png") for i in img_path_lis],
            train=False
        )


if __name__ == "__main__":
    main()
