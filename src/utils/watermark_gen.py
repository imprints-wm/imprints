import cv2
import os
import random
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from .image_process import load_image

FONT_PATH = "/usr/share/fonts/truetype"


def gen_black_bg(path="./", name="black_bg.png"):
    a = np.full((3, 256, 256), 0.0)
    a = np.transpose(a, (1, 2, 0)) * 255
    cv2.imwrite(path + name, a)

# Generate the absolute path list of all <file_type> files in the path
def all_files(path=FONT_PATH, file_type=".ttf"):  
    f_list = []

    def files_list(father_path):
        sub_path = os.listdir(father_path)  
        for sp in sub_path:
            full_sub_path = os.path.join(father_path, sp)  
            if os.path.isfile(full_sub_path):  
                file_name, post_name = os.path.splitext(full_sub_path)  
                if post_name == file_type:
                    f_list.append(file_name + post_name)
            else:  
                files_list(full_sub_path)

    files_list(path)
    return f_list   


def draw_text(text="usslab", size=25, save_path="./", save_name="txt_wm.png"):
    if not os.path.exists("./black_bg.png"):
        gen_black_bg()
    image = Image.open("./black_bg.png")
    draw = ImageDraw.Draw(image)
    font_lis = all_files()
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', size)

    width, height = draw.textsize(text, font)
    print(width, height)
    draw.text(((256-width)//2, (256-height)//2), text, font=font, align="left", stroke_width=0, fill=(204, 204, 204))
    
    image.save(save_path + save_name)


def gen_wm(text="S&P", text_size=42, standard_norm=False, device=torch.device("cuda:0")):
    image_size = 256

    draw_text(text=text, size=text_size)


    mask = (load_image(path="./txt_wm.png", gray=True)[1]>0.3).type(torch.float32).to(device)
    
    
    patch = cv2.imread("./txt_wm.png")
    patch = (torch.from_numpy(patch)/255.0).permute(2,0,1).to(device)
    print(patch.size())

    return patch[[2,1,0], :, :], mask

def load_logo(logo_path, standard_norm=False, device=torch.device("cuda:0")):
    image_size = 256
    scale_ratio = 0.5
    image = np.zeros([image_size, image_size, 4])

    
    patch = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    width = int(image_size*scale_ratio)
    patch_scaled = preprocess_logo(patch, width, width)

    padding = (image_size-width) // 2
    image[padding:padding+width, padding:padding+width, :] = patch_scaled

    patch_rgb = (torch.from_numpy(image[:,:,:3])/255.0).permute(2,0,1).type(torch.float32).to(device)
    
    if os.path.basename(logo_path) in ['6.png', '38.png']:
        mask = (torch.from_numpy((image[:,:,:3].sum(axis=2, keepdims=True)>20)*1.0)).permute(2,0,1).type(torch.float32).to(device)
    else:   
        mask = (torch.from_numpy(image[:,:,3:])/255.0).permute(2,0,1).type(torch.float32).to(device)
    
    patch_rgb = patch_rgb*mask
    return patch_rgb[[2,1,0], :, :], mask


def preprocess_logo(im, target_height, target_width):
    height, width = im.shape[:2]  

    ratio_h = height / target_height
    ration_w = width / target_width

    ratio = max(ratio_h, ration_w)


    size = (int(width / ratio), int(height / ratio))
    shrink = cv2.resize(im, size, interpolation=cv2.INTER_AREA)  
    BLACK = [0, 0, 0]  

    a = (target_width - int(width / ratio)) / 2
    b = (target_height - int(height / ratio)) / 2

    constant = cv2.copyMakeBorder(shrink, int(b), int(b), int(a), int(a), cv2.BORDER_CONSTANT, value=BLACK)
    constant = cv2.resize(constant, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return constant

