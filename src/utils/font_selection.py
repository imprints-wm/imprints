import cv2
import os
import random
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from image_process import load_image

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


def draw_text(text="usslab", size=35, save_path="./"):

    font_lis = all_files()

    for font_file in font_lis:
        try:
            if not os.path.exists("./black_bg.png"):
                gen_black_bg()
            image = Image.open("./black_bg.png")
            draw = ImageDraw.Draw(image)

            font = ImageFont.truetype(font_file, size)
        
            length = draw.textlength(text, font)
            
            draw.text(((255-length)//2, 90), text, font=font, align="left", stroke_width=0, fill=(104, 120, 146))
            
            save_name = os.path.basename(font_file)[:-4]+'.png'
            image.save(save_path + save_name)
        except Exception as e:
            print(e)


draw_text('ABCDEFGHIJK')
