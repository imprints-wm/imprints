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


def all_files(path=FONT_PATH, file_type=".ttf"):  # 生成path路径下全部file_type类型文件绝对路径列表
    f_list = []

    def files_list(father_path):
        sub_path = os.listdir(father_path)  # 读取父路径下全部文件或文件夹名称
        for sp in sub_path:
            full_sub_path = os.path.join(father_path, sp)  # 生成完整子路径
            if os.path.isfile(full_sub_path):  # 判断是否为文件
                file_name, post_name = os.path.splitext(full_sub_path)  # 获取文件后缀名
                if post_name == file_type:
                    f_list.append(file_name + post_name)
            else:  # 如果是文件夹，递归调用
                files_list(full_sub_path)

    files_list(path)
    return f_list  # 返回路径列表


def draw_text(text="usslab", size=35, save_path="./"):

    font_lis = all_files()
    # font = ImageFont.truetype(font_lis[0], size)
    for font_file in font_lis:
        try:
            if not os.path.exists("./black_bg.png"):
                gen_black_bg()
            image = Image.open("./black_bg.png")
            draw = ImageDraw.Draw(image)

            font = ImageFont.truetype(font_file, size)
            # stroke_width: The width of the text stroke. Default: 0.
            # TODO: make stroke_width an input parameter.
            length = draw.textlength(text, font)
            # print(length)
            draw.text(((255-length)//2, 90), text, font=font, align="left", stroke_width=0, fill=(104, 120, 146))
            # draw.text(((255-length)//2, 90), text, font=font, align="left", stroke_width=1, fill=(128+64+16, 128, 128))
            save_name = os.path.basename(font_file)[:-4]+'.png'
            image.save(save_path + save_name)
        except Exception as e:
            print(e)


draw_text('ABCDEFGHIJK')
