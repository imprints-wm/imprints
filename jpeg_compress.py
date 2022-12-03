import cv2
from PIL import Image
import torch
from torchvision.io import write_jpeg, encode_jpeg, decode_jpeg, ImageReadMode
from tqdm import tqdm
import os, shutil
import argparse


def convert_jpeg(filepath, savepath, quality):
    im = torch.from_numpy(cv2.imread(filepath)).permute(2,0,1)
    im = decode_jpeg(encode_jpeg(im, quality), mode=ImageReadMode.UNCHANGED)
    # write_jpeg(im[[2,1,0], :, :], savepath, quality)
    cv2.imwrite(savepath, im.permute(1,2,0).numpy())

def convert_all(png_dir, jpg_dir, quality):
    for png in tqdm(os.listdir(png_dir)):
        filepath = os.path.join(png_dir, png)
        savepath = os.path.join(jpg_dir, png)
        convert_jpeg(filepath, savepath, quality)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--png', type=str, help='path to png images')
    argparser.add_argument('--quality', type=int, help='compression quality')
    
    args = argparser.parse_args()

    if args.png[-1] == '/':
        args.png = args.png[:-1]

    jpg = args.png + "_j{:d}".format(args.quality)

    subdirs = ["Mask", "Watermark", "Watermark_free_image"]
    os.mkdir(jpg)
    os.mkdir(os.path.join(jpg, "Watermarked_image"))
    for sd in tqdm(subdirs):
        shutil.copytree(os.path.join(args.png, sd), 
                        os.path.join(jpg, sd))

    convert_all(os.path.join(args.png, "Watermarked_image"), 
                os.path.join(jpg, "Watermarked_image"), 
                args.quality)